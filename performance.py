"""
Performance Optimization System for FinViz Dashboard
Provides caching, data processing optimization, and performance monitoring
"""

import time
import functools
import hashlib
import pickle
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
import streamlit as st
from pathlib import Path
import logging
import psutil
import gc
from dataclasses import dataclass
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    cache_hit_ratio: float
    data_size_mb: float
    timestamp: datetime

class MemoryCache:
    """In-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            entry = self._cache[key]
            if datetime.now() > entry['expires_at']:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            # Evict if at max size
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
            self._access_times[key] = datetime.now()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'keys': list(self._cache.keys())
            }

class RedisCache:
    """Redis-based cache implementation"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.prefix = "finviz:"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            value = self.redis_client.get(f"{self.prefix}{key}")
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache"""
        try:
            serialized_value = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(f"{self.prefix}{key}", ttl, serialized_value)
            else:
                self.redis_client.set(f"{self.prefix}{key}", serialized_value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries with prefix"""
        try:
            keys = self.redis_client.keys(f"{self.prefix}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_cache = MemoryCache()
        self.redis_cache = None
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Initialize Redis cache if available and enabled
        if REDIS_AVAILABLE and self.config.cache.enable_redis:
            try:
                self.redis_cache = RedisCache(
                    host=self.config.cache.redis_host,
                    port=self.config.cache.redis_port,
                    db=self.config.cache.redis_db
                )
            except Exception as e:
                logger.warning(f"Could not initialize Redis cache: {e}")
    
    def cache_result(self, ttl: Optional[int] = None, use_redis: bool = False):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cache = self.redis_cache if use_redis and self.redis_cache else self.memory_cache
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache the result
                cache.set(cache_key, result, ttl or self.config.cache.cache_ttl_seconds)
                
                logger.debug(f"Cache miss for {func.__name__}, execution time: {execution_time:.2f}s")
                return result
            
            return wrapper
        return decorator
    
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator for monitoring function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get initial metrics
            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = end_memory - start_memory
                cpu_percent = process.cpu_percent()
                
                # Calculate data size if result is DataFrame
                data_size_mb = 0
                if isinstance(result, pd.DataFrame):
                    data_size_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
                
                # Calculate cache hit ratio
                cache_stats = self.memory_cache.stats()
                cache_hit_ratio = 0.0  # Simplified for now
                
                # Store metrics
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_percent=cpu_percent,
                    cache_hit_ratio=cache_hit_ratio,
                    data_size_mb=data_size_mb,
                    timestamp=datetime.now()
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Log performance if slow
                if execution_time > 1.0:
                    logger.warning(f"Slow function {func.__name__}: {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in monitored function {func.__name__}: {e}")
                raise
        
        return wrapper
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize data types
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                # Convert string columns to category if beneficial
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame memory reduced by {reduction:.1f}% ({original_memory/1024/1024:.1f}MB → {optimized_memory/1024/1024:.1f}MB)")
        
        return df
    
    def batch_process_data(self, data: pd.DataFrame, func: Callable, batch_size: int = 10000) -> pd.DataFrame:
        """Process large DataFrame in batches to manage memory"""
        if len(data) <= batch_size:
            return func(data)
        
        results = []
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            
            batch = data.iloc[start_idx:end_idx]
            batch_result = func(batch)
            results.append(batch_result)
            
            # Force garbage collection after each batch
            gc.collect()
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def parallel_process(self, data: List[Any], func: Callable, max_workers: Optional[int] = None) -> List[Any]:
        """Process data in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if max_workers is None:
            max_workers = min(4, len(data))
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in data}
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
        
        return results
    
    def get_performance_dashboard(self) -> None:
        """Display performance dashboard in Streamlit"""
        st.subheader("⚡ Performance Dashboard")
        
        if not self.metrics_history:
            st.info("No performance data available yet")
            return
        
        # Recent metrics
        recent_metrics = self.metrics_history[-10:]
        
        # Create metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
            st.metric("Avg Execution Time", f"{avg_execution_time:.2f}s")
        
        with col2:
            avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
            st.metric("Avg Memory Usage", f"{avg_memory_usage:.1f}MB")
        
        with col3:
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            st.metric("Avg CPU Usage", f"{avg_cpu:.1f}%")
        
        with col4:
            cache_stats = self.memory_cache.stats()
            st.metric("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
        
        # Performance trends
        if len(self.metrics_history) > 1:
            self._plot_performance_trends()
        
        # System information
        self._display_system_info()
    
    def _plot_performance_trends(self) -> None:
        """Plot performance trends"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        timestamps = [m.timestamp for m in self.metrics_history]
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Execution Time Trend", "Memory Usage Trend"),
            vertical_spacing=0.1
        )
        
        # Execution time trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=execution_times,
                mode='lines+markers',
                name='Execution Time',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        
        # Memory usage trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode='lines+markers',
                name='Memory Usage',
                line=dict(color='#ff7f0e')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title="Performance Trends"
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Seconds", row=1, col=1)
        fig.update_yaxes(title_text="MB", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_system_info(self) -> None:
        """Display system information"""
        st.subheader("🖥️ System Information")
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**CPU Information**")
            st.write(f"• CPU Cores: {cpu_count}")
            st.write(f"• CPU Usage: {psutil.cpu_percent()}%")
            
            st.write("**Memory Information**")
            st.write(f"• Total Memory: {memory.total / 1024**3:.1f} GB")
            st.write(f"• Available Memory: {memory.available / 1024**3:.1f} GB")
            st.write(f"• Memory Usage: {memory.percent}%")
        
        with col2:
            st.write("**Disk Information**")
            st.write(f"• Total Disk: {disk.total / 1024**3:.1f} GB")
            st.write(f"• Free Disk: {disk.free / 1024**3:.1f} GB")
            st.write(f"• Disk Usage: {disk.percent}%")
            
            st.write("**Cache Information**")
            cache_stats = self.memory_cache.stats()
            st.write(f"• Memory Cache: {cache_stats['size']}/{cache_stats['max_size']}")
            if self.redis_cache:
                st.write("• Redis Cache: Enabled")
            else:
                st.write("• Redis Cache: Disabled")
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        info = {
            'memory_cache': self.memory_cache.stats(),
            'redis_cache_enabled': self.redis_cache is not None
        }
        return info
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call"""
        # Create a hash of the function name and arguments
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

# Specialized optimization functions for financial data
class FinancialDataOptimizer:
    """Specialized optimizer for financial data processing"""
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        self.optimizer = performance_optimizer
    
    @staticmethod
    def optimize_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize transaction DataFrame for better performance"""
        if df.empty:
            return df
        
        # Convert date columns to datetime with optimal format
        date_columns = ['Date', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Optimize numeric columns
        numeric_columns = ['Deposits', 'Withdrawls', 'Balance', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Optimize categorical columns
        categorical_columns = ['Category', 'transaction_type', 'account_name']
        for col in categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
        
        return df
    
    def create_indexed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create properly indexed DataFrame for faster queries"""
        if df.empty or 'Date' not in df.columns:
            return df
        
        # Set date as index if it's not already
        if df.index.name != 'Date':
            df = df.set_index('Date')
        
        # Sort by index for optimal range queries
        df = df.sort_index()
        
        return df
    
    def aggregate_monthly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-aggregate monthly data for faster visualization"""
        if df.empty or 'Date' not in df.columns:
            return pd.DataFrame()
        
        # Ensure Date is datetime
        if df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by month and aggregate
        monthly_agg = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Deposits': 'sum',
            'Withdrawls': 'sum',
            'Balance': 'last',
            'Description': 'count'  # Transaction count
        }).rename(columns={'Description': 'transaction_count'})
        
        monthly_agg['net_flow'] = monthly_agg['Deposits'] - monthly_agg['Withdrawls']
        
        return monthly_agg

# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance (singleton)"""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    
    return _performance_optimizer

def cache_result(ttl: Optional[int] = None, use_redis: bool = False):
    """Decorator for caching function results"""
    optimizer = get_performance_optimizer()
    return optimizer.cache_result(ttl=ttl, use_redis=use_redis)

def monitor_performance(func: Callable) -> Callable:
    """Decorator for monitoring function performance"""
    optimizer = get_performance_optimizer()
    return optimizer.monitor_performance(func)

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage"""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_dataframe(df)

# Streamlit performance utilities
def show_performance_metrics():
    """Show performance metrics in Streamlit sidebar"""
    optimizer = get_performance_optimizer()
    
    with st.sidebar:
        st.markdown("---")
        if st.button("⚡ Performance", key="perf_toggle"):
            st.session_state.show_performance = not st.session_state.get("show_performance", False)
        
        if st.session_state.get("show_performance", False):
            optimizer.get_performance_dashboard()
            
            if st.button("🗑️ Clear Cache"):
                optimizer.clear_cache()
                st.success("Cache cleared!")

@st.cache_data(ttl=3600)
def cached_data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """Cached data processing for Streamlit"""
    optimizer = FinancialDataOptimizer(get_performance_optimizer())
    
    # Optimize the data
    optimized_data = optimizer.optimize_transaction_data(data)
    optimized_data = optimizer.create_indexed_dataframe(optimized_data)
    
    return optimized_data 