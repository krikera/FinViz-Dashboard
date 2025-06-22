import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
from utils import nlp_engine, enhanced_transaction_analysis

def safe_get_descriptions(data):
    """Safely extract descriptions handling categorical data types"""
    try:
        # Convert categorical to string if needed
        if data['Description'].dtype.name == 'category':
            descriptions = data['Description'].astype(str).fillna('Unknown Transaction').tolist()
        else:
            descriptions = data['Description'].fillna('Unknown Transaction').tolist()
        return descriptions
    except Exception as e:
        st.error(f"Error processing descriptions: {e}")
        return ['Unknown Transaction'] * len(data)

def display_nlp_dashboard(data):
    """Main NLP dashboard function that displays comprehensive NLP analysis"""
    st.subheader("🧠 Advanced NLP Analysis Dashboard")
    
    if not nlp_engine or not nlp_engine.nlp_available:
        st.warning("NLP engine not available. Please ensure all NLP dependencies are installed.")
        st.code("pip install transformers torch spacy nltk textblob wordcloud gensim sentence-transformers")
        st.code("python -m spacy download en_core_web_sm")
        return
    
    if data.empty:
        st.warning("No data available for NLP analysis")
        return
    
    # Create tabs for different NLP analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "💭 Sentiment Analysis", 
        "🏷️ Smart Categorization", 
        "☁️ Word Cloud", 
        "🔍 Insights & Patterns"
    ])
    
    # Safely extract descriptions
    descriptions = safe_get_descriptions(data)
    
    with tab1:
        display_nlp_overview(data, descriptions)
    
    with tab2:
        display_sentiment_analysis(data, descriptions)
    
    with tab3:
        display_smart_categorization(data, descriptions)
    
    with tab4:
        display_wordcloud_analysis(descriptions)
    
    with tab5:
        display_insights_and_patterns(data, descriptions)

def display_nlp_overview(data, descriptions):
    """Display NLP analysis overview"""
    st.write("### 📊 NLP Analysis Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(descriptions))
    
    with col2:
        avg_length = np.mean([len(str(desc).split()) for desc in descriptions if desc and str(desc) != 'nan'])
        st.metric("Avg Words per Description", f"{avg_length:.1f}")
    
    with col3:
        unique_merchants = len(set([str(desc).split()[0] if desc and str(desc) != 'nan' and str(desc).split() else "Unknown" for desc in descriptions]))
        st.metric("Unique Starting Words", unique_merchants)
    
    with col4:
        try:
            analysis = enhanced_transaction_analysis(descriptions)
            if analysis and analysis.get('anomalies'):
                anomaly_count = sum(analysis['anomalies'])
                st.metric("Anomalous Descriptions", anomaly_count)
            else:
                st.metric("Anomalous Descriptions", "N/A")
        except:
            st.metric("Anomalous Descriptions", "N/A")
    
    # Transaction length distribution
    st.write("### Description Length Distribution")
    lengths = [len(str(desc).split()) for desc in descriptions if desc and str(desc) != 'nan']
    
    if lengths:
        fig = px.histogram(
            x=lengths, 
            nbins=20,
            title="Distribution of Words per Transaction Description",
            labels={'x': 'Number of Words', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid descriptions found for analysis")

def display_sentiment_analysis(data, descriptions):
    """Display sentiment analysis results"""
    st.write("### 💭 Transaction Sentiment Analysis")
    
    try:
        # Filter out empty descriptions
        valid_descriptions = [desc for desc in descriptions if desc and str(desc) != 'nan' and str(desc).strip()]
        
        if not valid_descriptions:
            st.warning("No valid descriptions available for sentiment analysis")
            return
        
        # Get sentiment analysis
        sentiment_results = nlp_engine.sentiment_analysis(valid_descriptions)
        
        if not sentiment_results:
            st.warning("Sentiment analysis not available")
            return
        
        # Create a copy of data for sentiment analysis
        data_with_sentiment = data.copy()
        
        # Ensure we have the right number of sentiment results
        if len(sentiment_results) != len(data):
            # Pad or truncate sentiment results to match data length
            sentiment_mapping = {}
            for i, desc in enumerate(descriptions):
                if desc and str(desc) != 'nan' and str(desc).strip():
                    # Find corresponding sentiment result
                    valid_index = sum(1 for d in descriptions[:i] if d and str(d) != 'nan' and str(d).strip())
                    if valid_index < len(sentiment_results):
                        sentiment_mapping[i] = sentiment_results[valid_index]
                    else:
                        sentiment_mapping[i] = {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'polarity': 0.0}
                else:
                    sentiment_mapping[i] = {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'polarity': 0.0}
            
            # Apply sentiment results
            data_with_sentiment['Sentiment'] = [sentiment_mapping.get(i, {'sentiment': 'NEUTRAL'})['sentiment'] for i in range(len(data))]
            data_with_sentiment['Sentiment_Confidence'] = [sentiment_mapping.get(i, {'confidence': 0.0})['confidence'] for i in range(len(data))]
            data_with_sentiment['Polarity'] = [sentiment_mapping.get(i, {'polarity': 0.0}).get('polarity', 0.0) for i in range(len(data))]
        else:
            # Direct mapping
            data_with_sentiment['Sentiment'] = [result.get('sentiment', 'NEUTRAL') for result in sentiment_results]
            data_with_sentiment['Sentiment_Confidence'] = [result.get('confidence', 0.0) for result in sentiment_results]
            data_with_sentiment['Polarity'] = [result.get('polarity', 0.0) for result in sentiment_results]
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = data_with_sentiment['Sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment by amount
            if 'Withdrawls' in data_with_sentiment.columns:
                fig = px.box(
                    data_with_sentiment,
                    x='Sentiment',
                    y='Withdrawls',
                    title="Withdrawal Amounts by Sentiment"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        if 'Date' in data_with_sentiment.columns:
            data_with_sentiment['Date'] = pd.to_datetime(data_with_sentiment['Date'])
            daily_sentiment = data_with_sentiment.groupby([
                data_with_sentiment['Date'].dt.date, 'Sentiment'
            ]).size().reset_index(name='count')
            
            fig = px.bar(
                daily_sentiment,
                x='Date',
                y='count',
                color='Sentiment',
                title="Sentiment Trends Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show sample transactions by sentiment
        st.write("### Sample Transactions by Sentiment")
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_data = data_with_sentiment[data_with_sentiment['Sentiment'] == sentiment]
            if not sentiment_data.empty:
                st.write(f"**{sentiment} Transactions:**")
                sample_size = min(3, len(sentiment_data))
                sample_transactions = sentiment_data.nlargest(sample_size, 'Sentiment_Confidence')
                for _, row in sample_transactions.iterrows():
                    desc = str(row['Description']) if pd.notna(row['Description']) else 'Unknown Transaction'
                    st.write(f"- {desc} (Confidence: {row['Sentiment_Confidence']:.2f})")
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        st.write("Debug info:", str(e))

def display_smart_categorization(data, descriptions):
    """Display advanced categorization results"""
    st.write("### 🏷️ Smart Transaction Categorization")
    
    try:
        # Filter out empty descriptions
        valid_descriptions = [desc for desc in descriptions if desc and str(desc) != 'nan' and str(desc).strip()]
        
        if not valid_descriptions:
            st.warning("No valid descriptions available for categorization")
            return
        
        # Get categorization results
        categorization_results = nlp_engine.advanced_transaction_categorization(valid_descriptions)
        
        if not categorization_results:
            st.warning("Smart categorization not available")
            return
        
        # Create a copy of data for categorization
        data_with_categories = data.copy()
        
        # Map categorization results back to original data
        if len(categorization_results) != len(data):
            category_mapping = {}
            for i, desc in enumerate(descriptions):
                if desc and str(desc) != 'nan' and str(desc).strip():
                    valid_index = sum(1 for d in descriptions[:i] if d and str(d) != 'nan' and str(d).strip())
                    if valid_index < len(categorization_results):
                        category_mapping[i] = categorization_results[valid_index]
                    else:
                        category_mapping[i] = {'category': 'other', 'confidence': 0.0}
                else:
                    category_mapping[i] = {'category': 'other', 'confidence': 0.0}
            
            data_with_categories['NLP_Category'] = [category_mapping.get(i, {'category': 'other'})['category'] for i in range(len(data))]
            data_with_categories['Category_Confidence'] = [category_mapping.get(i, {'confidence': 0.0})['confidence'] for i in range(len(data))]
        else:
            data_with_categories['NLP_Category'] = [result.get('category', 'other') for result in categorization_results]
            data_with_categories['Category_Confidence'] = [result.get('confidence', 0.0) for result in categorization_results]
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = data_with_categories['NLP_Category'].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Smart Category Distribution"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Spending by category
            if 'Withdrawls' in data_with_categories.columns:
                category_spending = data_with_categories.groupby('NLP_Category')['Withdrawls'].sum().sort_values(ascending=False)
                fig = px.bar(
                    x=category_spending.index,
                    y=category_spending.values,
                    title="Total Spending by Smart Category"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.write("### Categorization Confidence")
        fig = px.histogram(
            data_with_categories,
            x='Category_Confidence',
            title="Distribution of Categorization Confidence Scores",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show comparison with original categories if available
        if 'Category' in data.columns:
            st.write("### Smart vs Original Categories")
            comparison = data_with_categories.groupby(['Category', 'NLP_Category']).size().reset_index(name='count')
            if not comparison.empty:
                pivot_table = comparison.pivot(index='Category', columns='NLP_Category', values='count').fillna(0)
                
                fig = px.imshow(
                    pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    title="Original vs Smart Categories Heatmap",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in smart categorization: {e}")

def display_wordcloud_analysis(descriptions):
    """Display word cloud and text analysis"""
    st.write("### ☁️ Word Cloud & Text Analysis")
    
    try:
        # Filter valid descriptions
        valid_descriptions = [str(desc) for desc in descriptions if desc and str(desc) != 'nan' and str(desc).strip()]
        
        if not valid_descriptions:
            st.warning("No valid descriptions available for word cloud analysis")
            return
        
        # Generate word cloud
        if nlp_engine and hasattr(nlp_engine, 'create_wordcloud'):
            try:
                wordcloud = nlp_engine.create_wordcloud(valid_descriptions)
                
                if wordcloud:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Could not generate word cloud")
            except Exception as e:
                st.warning(f"Word cloud generation failed: {e}")
        
        # Top words analysis
        all_text = ' '.join([desc.lower() for desc in valid_descriptions])
        words = all_text.split()
        
        # Filter out common stop words and short words
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        if filtered_words:
            word_counts = Counter(filtered_words)
            top_words = word_counts.most_common(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Top 20 Words")
                words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                fig = px.bar(
                    words_df,
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title="Most Frequent Words"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("#### Word Frequency Distribution")
                frequencies = list(word_counts.values())
                fig = px.histogram(
                    x=frequencies,
                    title="Word Frequency Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No meaningful words found for analysis")
        
    except Exception as e:
        st.error(f"Error in word cloud analysis: {e}")

def display_insights_and_patterns(data, descriptions):
    """Display financial insights and patterns"""
    st.write("### 🔍 Financial Insights & Patterns")
    
    try:
        # Filter valid descriptions
        valid_descriptions = [str(desc) for desc in descriptions if desc and str(desc) != 'nan' and str(desc).strip()]
        
        if not valid_descriptions:
            st.warning("No valid descriptions available for insights analysis")
            return
        
        # Extract financial insights
        insights = nlp_engine.extract_financial_insights(valid_descriptions)
        
        if not insights:
            st.warning("Financial insights not available")
            return
        
        # Display insights in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if insights.get('merchants'):
                st.write("#### 🏪 Detected Merchants")
                merchant_counts = Counter(insights['merchants'])
                top_merchants = merchant_counts.most_common(10)
                for merchant, count in top_merchants:
                    st.write(f"- {merchant}: {count} transactions")
            
            if insights.get('locations'):
                st.write("#### 📍 Detected Locations")
                location_counts = Counter(insights['locations'])
                top_locations = location_counts.most_common(5)
                for location, count in top_locations:
                    st.write(f"- {location}: {count} mentions")
        
        with col2:
            if insights.get('financial_terms'):
                st.write("#### 💰 Financial Terms")
                term_counts = Counter(insights['financial_terms'])
                top_terms = term_counts.most_common(10)
                for term, count in top_terms:
                    st.write(f"- {term}: {count} times")
            
            if insights.get('spending_themes'):
                st.write("#### 🎯 Spending Themes")
                for i, theme in enumerate(insights['spending_themes'][:5], 1):
                    st.write(f"{i}. {theme}")
        
        # Spending patterns visualization
        if insights.get('merchants') and 'Withdrawls' in data.columns:
            st.write("### Merchant Spending Analysis")
            
            # Create merchant spending data
            merchant_data = []
            for _, row in data.iterrows():
                desc = str(row['Description']).lower() if pd.notna(row['Description']) else ''
                for merchant in insights['merchants']:
                    if merchant.lower() in desc:
                        merchant_data.append({
                            'Merchant': merchant,
                            'Amount': row.get('Withdrawls', 0),
                            'Date': row.get('Date')
                        })
            
            if merchant_data:
                merchant_df = pd.DataFrame(merchant_data)
                merchant_spending = merchant_df.groupby('Merchant')['Amount'].agg(['sum', 'count']).reset_index()
                merchant_spending.columns = ['Merchant', 'Total_Spending', 'Transaction_Count']
                
                fig = px.scatter(
                    merchant_spending,
                    x='Transaction_Count',
                    y='Total_Spending',
                    size='Total_Spending',
                    hover_name='Merchant',
                    title="Merchant Analysis: Frequency vs Total Spending"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection results
        try:
            analysis = enhanced_transaction_analysis(valid_descriptions)
            if analysis and analysis.get('anomalies'):
                st.write("### 🚨 Anomalous Transaction Patterns")
                anomalous_indices = [i for i, is_anomaly in enumerate(analysis['anomalies']) if is_anomaly]
                
                if anomalous_indices:
                    st.write(f"Found {len(anomalous_indices)} potentially anomalous transactions:")
                    # Map back to original data indices
                    original_anomalous_indices = []
                    valid_index = 0
                    for i, desc in enumerate(descriptions):
                        if desc and str(desc) != 'nan' and str(desc).strip():
                            if valid_index in anomalous_indices:
                                original_anomalous_indices.append(i)
                            valid_index += 1
                    
                    if original_anomalous_indices:
                        anomalous_data = data.iloc[original_anomalous_indices]
                        
                        # Show top anomalous transactions
                        display_count = min(5, len(anomalous_data))
                        st.write(f"**Top {display_count} Anomalous Transactions:**")
                        for _, row in anomalous_data.head(display_count).iterrows():
                            desc = str(row['Description']) if pd.notna(row['Description']) else 'Unknown Transaction'
                            amount = row.get('Withdrawls', 0)
                            st.write(f"- {desc} | Amount: ${amount:.2f}")
                else:
                    st.write("No anomalous transactions detected.")
        except Exception as e:
            st.warning(f"Anomaly detection not available: {e}")
        
    except Exception as e:
        st.error(f"Error in insights analysis: {e}")

def create_nlp_summary_report(data):
    """Create a comprehensive NLP summary report"""
    if not nlp_engine or not nlp_engine.nlp_available:
        return "NLP analysis not available"
    
    try:
        descriptions = safe_get_descriptions(data)
        
        # Generate comprehensive summary
        summary = nlp_engine.generate_spending_summary(data)
        
        # Add additional insights
        valid_descriptions = [desc for desc in descriptions if desc and str(desc) != 'nan' and str(desc).strip()]
        analysis = enhanced_transaction_analysis(valid_descriptions)
        if analysis:
            summary += "\n\n**Additional Insights:**\n"
            
            # Sentiment summary
            if analysis.get('sentiment'):
                positive_count = sum(1 for s in analysis['sentiment'] if s.get('sentiment') == 'POSITIVE')
                negative_count = sum(1 for s in analysis['sentiment'] if s.get('sentiment') == 'NEGATIVE')
                summary += f"- Sentiment: {positive_count} positive, {negative_count} negative transactions\n"
            
            # Category distribution
            if analysis.get('categorization'):
                categories = [c.get('category', 'Unknown') for c in analysis['categorization']]
                top_category = Counter(categories).most_common(1)[0] if categories else None
                if top_category:
                    summary += f"- Most common spending category: {top_category[0]} ({top_category[1]} transactions)\n"
        
        return summary
        
    except Exception as e:
        return f"Error generating NLP summary: {e}" 