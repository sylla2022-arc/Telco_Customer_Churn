import requests
import streamlit as st
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Échantillon de données
df_brute = pd.read_csv('data_churn/Customer_Churn.csv')
if not os.path.exists("data_churn/sample_clients.csv"):
    df_sample = df_brute.sample(n=40, random_state=42)
    df_sample.to_csv("data_churn/sample_clients.csv", index=False, header=True)

def main_app():
    # Configuration de la page
    st.set_page_config(page_title="Telco Customer Churn", layout="wide")


    st.sidebar.title("Telco Customer Churn Prediction and Monitoring")
    st.sidebar.write("**Auteur:** M. SYLLA")
    st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/mahamadou-sylla/)")
    
    pages = ['Contexte', 'Test Demo', 'Prediction', 'Monitoring', 'Conclusion']
    selected_page = st.sidebar.radio('Choississez une option:', pages)
    
    # Session state initialization
    if "sample" not in st.session_state:
        st.session_state.sample = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    # Page: Contexte
    if selected_page == 'Contexte':
        st.markdown("""
        #### Problématique : Pourquoi vous perdez vos clients ?
        
        80% des clients quittent l'entreprise dans les 6 mois suivant leur souscription.
        Cela représente un coût important en termes de revenus perdus et de dépenses marketing
        pour acquérir de nouveaux clients.
        
        Il est donc crucial de comprendre les raisons de ce churn et d'anticiper les départs pour 
        mettre en place des actions correctives.
        
        Dans ce projet, j'ai décidé de vous révéler les facteurs clés qui influencent le churn 
        des clients et comment les monitorer efficacement.
        """)

    # Page: Test Demo
    elif selected_page == 'Test Demo':
        st.header("Test Démo - Prédiction sur échantillon")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Charger un échantillon de données", use_container_width=True):
                st.session_state.sample = pd.read_csv("data_churn/sample_clients.csv")
                st.success("Fichier chargé avec succès")
                
            if st.session_state.sample is not None:
                st.dataframe(st.session_state.sample.head(6))
                st.info(f"Échantillon de {len(st.session_state.sample)} clients chargé")
        
        with col2:
            if st.button("🚀 Lancer la prédiction", use_container_width=True):
                if st.session_state.sample is not None:
                    try:
                        API_URL = "http://localhost:8000/predict"
                        files = {"file": st.session_state.sample.to_csv(index=False).encode("utf-8")}
                        response = requests.post(API_URL, files=files)
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            predictions = response_data["preview"]
                            st.session_state.prediction = pd.DataFrame(predictions)
                            
                            # Afficher le type de prédiction
                            if response_data.get("is_global", False):
                                st.success("✅ Prédictions GLOBALES générées avec succès")
                            else:
                                st.success("✅ Prédictions d'ÉCHANTILLON générées avec succès")
                            
                            st.dataframe(st.session_state.prediction.head())
                            
                            # Afficher des informations supplémentaires
                            st.info(f"Nombre de clients prédits: {len(st.session_state.prediction)}")
                            
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                    except Exception as e:
                        st.error(f"Erreur de connexion: {str(e)}")
                else:
                    st.warning("Veuillez charger un échantillon avant de lancer la prédiction.")

        # Ajouter un bouton pour les prédictions globales
        if st.button("Générer des prédictions GLOBALES", use_container_width=True):
            try:
                df_brute = pd.read_csv('data_churn/Customer_Churn.csv')
                API_URL = "http://localhost:8000/predict"
                files = {"file": df_brute.to_csv(index=False).encode("utf-8")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    st.success("✅ Prédictions GLOBALES générées avec succès!")
                    st.info("Le monitoring global utilisera maintenant ces données complètes")
                else:
                    st.error(f"Erreur lors de la génération des prédictions globales: {response.status_code}")
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")

    # Page: Prediction
    elif selected_page == 'Prediction':
        st.header("Suivi des Prédictions")
        
        if st.button("🔄 Charger les dernières prédictions"):
            try:
                pred_files = [f for f in os.listdir("artifact/predict_data") if f.endswith('.csv')]
                if pred_files:
                    current_file = sorted(pred_files)[-1]
                    st.session_state.prediction = pd.read_csv(f"artifact/predict_data/{current_file}")
                    st.success(f"Prédictions chargées: {current_file}")
                else:
                    st.warning("Aucun fichier de prédiction trouvé")
            except Exception as e:
                st.error(f"Erreur lors du chargement: {str(e)}")
        
        if st.session_state.prediction is not None:
            st.dataframe(st.session_state.prediction.head(10))
            
            # Statistiques des prédictions
            if 'prediction' in st.session_state.prediction.columns:
                churn_count = st.session_state.prediction['prediction'].sum()
                total_count = len(st.session_state.prediction)
                churn_rate = (churn_count / total_count) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total clients", total_count)
                col2.metric("Churn prédit", churn_count)
                col3.metric("Taux de churn prédit", f"{churn_rate:.1f}%")

    # Page: Monitoring
    elif selected_page == 'Monitoring':
        st.header("Monitoring de la dérive de données")
        
        # Choix du type de monitoring
        monitoring_type = st.radio(
            "Type de monitoring:",
            ["Monitoring Global", "Monitoring sur Échantillon"],
            horizontal=True,
            key="monitoring_type"
        )
        
        # Initialiser response à None
        response = None
        response_perf = None
        URL_TARGET_drift = "http://localhost:8000/target-drift"
        API_URL_PREDICT_SAMPLE_DRIFT = "http://localhost:8000/predict-sample-drift"
        URL_PERFORMANCE_METRICS= "http://localhost:8000/performance-metrics"
        try:
            if monitoring_type == "Monitoring Global":
                # Monitoring global - données complètes
                response = requests.get(URL_TARGET_drift).json()
                response_perf = requests.get(URL_PERFORMANCE_METRICS).json()
                st.success("Monitoring global effectué avec succès")
                
            else:
                # Monitoring sur échantillon
                if st.session_state.sample is not None:
                    
                    files = {"file": st.session_state.sample.to_csv(index=False).encode("utf-8")}
                    response = requests.post(API_URL_PREDICT_SAMPLE_DRIFT, files=files).json()
                    st.success("Monitoring sur échantillon effectué avec succès")
                else:
                    st.warning("Veuillez d'abord charger un échantillon dans l'onglet 'Test Demo'")
                    return # On stop l'exécution

        except Exception as e:
            st.error(f"Erreur lors du monitoring: {str(e)}")
            st.info("Vérifiez que l'API FastAPI est en cours d'exécution sur http://localhost:8000")
        
    
                
        if response is not None:
            if monitoring_type == "Monitoring Global":
                # Affichage pour monitoring global
                nb_clients = response.get("nb_clients", 0)
                churn_rate = response.get('churn_rate', 0) * 100
                drift_detected = response.get('target_drift_detected', False)
                drift_score = response.get('drift_score', 0)

        
                col1, col2, col3 = st.columns(3)
                col1.metric("👥 Nombre de clients", f"{nb_clients:,}")
                col2.metric("📉 Taux de churn", f"{churn_rate:.1f}%")
                col3.metric(
                    "⚖️ Dérive détectée", 
                    "✅ Oui" if drift_detected else "❌ Non",
                    delta=f"{drift_score:.2f}" if drift_detected else None,
                    delta_color="inverse"
                )

                # perf global
                if response_perf is not None:
                    accuracy = response_perf.get("accuracy", 0)
                    recall = response_perf.get("recall", 0)
                    precision = response_perf.get("precision", 0)
                    f1_score_val = response_perf.get("f1_score", 0)
                    confusion_matrix = response_perf.get('confusion_matrix', [])
                    
                    # Performance model
                    st.subheader("📊 Performance du modèle")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        container = st.container(border=True)
                        with container:
                            st.write("**📈 Métriques**")
                            st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                            st.metric("🎯 Precision", f"{precision:.3f}")

                    with col2:
                        container = st.container(border=True)
                        with container:
                            st.write("**📈 Métriques**")
                            st.metric("🎯 Recall", f"{recall:.3f}")
                            st.metric("🎯 F1 Score", f"{f1_score_val:.3f}")

                    with col3:
                        container = st.container(border=True)
                        with container:
                            if confusion_matrix and len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2:
                                st.write("**🧮 Matrice de Confusion**")
                                
                                # Créer la heatmap avec Plotly
                                fig = px.imshow(
                                    confusion_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    x=['Prédit Non', 'Prédit Oui'],
                                    y=['Réel Non', 'Réel Oui'],
                                    color_continuous_scale='Blues'
                                )
                                
                                fig.update_layout(
                                    height=170,
                                    margin=dict(l=5, r=5, t=5, b=5),
                                    coloraxis_showscale=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Matrice non disponible")
                    
            else:
                # Affichage pour monitoring sur échantillon
                drift_detected = response.get("drift_detected", False)
                drift_score = response.get("drift_score", 0)
                nb_clients = response.get("nb_clients", 0)
                churn_rate = response.get('churn_rate', 0) * 100

                 # Echantillons
                col1, col2, col3 = st.columns(3)
                col1.metric("👥 Nombre de clients", f"{nb_clients:,}")
                col2.metric("📉 Taux de churn", f"{churn_rate:.1f}%")
                col3.metric(
                    "⚖️ Dérive détectée", 
                    "✅ Oui" if drift_detected else "❌ Non",
                    delta=f"{drift_score:.2f}" if drift_detected else None,
                    delta_color="inverse"
                )
  

            # Visualisations
            st.subheader("Distribution du Churn")
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
            
            # Pie chart
            churn_data = response.get('churn_distribution', {'Oui': 375, 'Non': 1125})
            #print(churn_data)
            fig.add_trace(go.Pie(
                labels=list(churn_data.keys()),
                values=list(churn_data.values()),
                name="Répartition"
            ), 1, 1)
            
            # Bar chart
            fig.add_trace(go.Bar(
                x=list(churn_data.keys()),
                y=list(churn_data.values()),
                marker_color=['red', 'green'],
                name="Distribution"
            ), 1, 2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Features importantes
            st.subheader("Analyse des variables importantes")
            
            features_data = response.get('important_features', {})
            if features_data:
                fig_features = go.Figure()
                
                for feature, values in features_data.items():
                    if isinstance(values, dict) and 'reference' in values and 'current' in values:
                        fig_features.add_trace(go.Scatter(
                            x=list(range(len(values['reference']))),
                            y=values['reference'],
                            name=f"{feature} (Ref)",
                            line=dict(dash='dash')
                        ))
                        fig_features.add_trace(go.Scatter(
                            x=list(range(len(values['current']))),
                            y=values['current'],
                            name=f"{feature} (Actuel)"
                        ))
                
                fig_features.update_layout(
                    title="Distribution des features importantes",
                    xaxis_title="Index",
                    yaxis_title="Valeur"
                )
                st.plotly_chart(fig_features, use_container_width=True)
            
                # Recommandations
                st.subheader("Recommandations")
                if drift_detected:
                    st.error("""
                    ⚠️ **Alerte : Dérive détectée**
                    - Vérifier la qualité des données d'entrée
                    - Recalibrer le modèle si nécessaire
                    - Analyser les changements dans le comportement des clients
                    """)
                else:
                    st.success("""
                    ✅ **Statut stable**
                    - Aucune dérive significative détectée
                    - Le modèle performe de manière cohérente
                    - Continuer le monitoring régulier
                    """)

            else:
                if monitoring_type == "Monitoring sur Échantillon" and st.session_state.sample is None:
                    pass
                else:
                    st.info("Sélectionnez un type de monitoring et cliquez pour lancer l'analyse")
        # page conclusion         
    elif selected_page == 'Conclusion':
        st.header("Conclusion")
        st.markdown("""
        ### Insights clés
        
        - **Taux de churn moyen**: 26% des clients quittent le service dans le cas général.
        - **Facteurs déclencheurs**: Ancienneté, type de service internet, charges mensuelles, Charges Totales
        - **Recommandations**:
          - Programmes de fidélisation pour les clients à risque (Promotion, revision de contrat, etc)
          - Offres personnalisées pour les clients fibre optique
          - Surveillance continue de la dérive des données pour detecter plutôt les clients sur la pente de decrochage.
    """)

if __name__ == '__main__':
    main_app()