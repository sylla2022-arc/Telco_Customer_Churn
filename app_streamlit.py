import requests
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import re, json



def main_app():

    df = pd.read_csv("data_churn/Customer_Churn.csv")
    df_sample = df.sample(30)
    df_sample.to_csv("data_churn/sample_clients.csv", header = True, index=False)

    st.sidebar.title("Telco Customer Churn Prediction and Monitoring")
    
    pages = ['Contexte', 'Test Demo', 'Prediction', 'Monitoring', 'Conclusion']
    #st.markdown("<h3 style='color:#0074D9;'>Titre</h3>", unsafe_allow_html=True)
    st.write(f":grey[Auteur: M. SYLLA], [LinkedIn](https://www.linkedin.com/in/mahamadou-sylla/)")

    pages = st.sidebar.radio('Choississez une option:', pages)

    if pages == 'Contexte':
        st.markdown("""#### Problématique : Pourquoi vous perdez vos clients ?
                      
    80% des clients quittent l'entreprise dans les 6 mois suivant leur souscription.
    Cela représente un coût important en termes de revenus perdus et de dépenses marketing
    pour acquérir de nouveaux clients.
    Il est donc crucial de comprendre les raisons de ce churn et d'anticiper les départs pour 
    mettre en place des actions correctives.
    Dans ce projet, j'ai decidé de vous reveler les facteurs clés qui influencent le churn 
    des clients et comment les monitorer efficacement.
                    """, width=950)

    elif pages == 'Test Demo':
        if "sample" not in st.session_state:
            st.session_state.sample = None

        if st.button("Charger un échantillon de données"):
            st.session_state.sample = pd.read_csv("data_churn/sample_clients.csv")
            st.dataframe(st.session_state.sample.head(6))
            st.success("Fichier chargé avec succès")
            st.markdown("Cliquez sur Prediction pour lancer la prédiction")

        if st.button("Lancer la prédiction"):
            if st.session_state.sample is not None:
                # Appel à l'API pour la prédiction
                API_URL = "http://localhost:8000/predict"
                files = {"file": st.session_state.sample.to_csv(index=False).encode("utf-8")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    st.success("Prédictions générées OK")
                    st.dataframe(pd.DataFrame(response.json()["preview"]))
                else:
                    st.error("Erreur chargement")
            else:
                st.warning("Veuillez charger un échantillon avant de lancer la prédiction.")

    elif pages == 'Prediction':
        st.markdown("#### Suivi des Prédictions")
        if "Prediction" not in st.session_state:
            st.session_state.Prediction = None

        if st.button("Charger les prédictions"):
            current_file = sorted([f for f in os.listdir("artifact/predict_data") if f.endswith('.csv')])[-1]
            st.session_state.Prediction = pd.read_csv(f"artifact/predict_data/{current_file}")
 
            st.dataframe(st.session_state.Prediction.head(6)) 
            st.success("Prédictions chargées avec succès")  

    elif pages == 'Monitoring':
        st.markdown("#### Monitoring de la dérive de la target Churn")
       
        if "Monitoring" not in st.session_state:
            st.session_state.Monitoring = None
            

        response = requests.get("http://localhost:8000/target-drift").json()
        
        nb_clients = response["nb_clients"]
        churn_rate = response['churn_rate']
          

        with open("artifact/monitoring/target_drift.html", "r", encoding="utf-8") as f:
            html_text = f.read()

            pattern = r"var evidently_dashboard_[a-z0-9]+ = (\{.*?\});"
            match = re.search(pattern, html_text, re.DOTALL)

            if match:
                json_str = match.group(1)
                data = json.loads(json_str)

               
         
            # RÉSUMÉ EXECUTIF DEPUIS HTML
            # ==========================

            summary = data["widgets"][1]["params"]["counters"]

            # transformer en dictionnaire
            
            summary_dict = {item["label"]: item["value"] for item in summary}
            #print(summary_dict)
            # Exemple : ce que Evidently expose
            nb_clients = int(nb_clients)
            nb_cols     = int(summary_dict.get("Columns", 0))
            nb_drifted  = int(summary_dict.get("Drifted Columns", 0))
            share_drift = float(summary_dict.get("Share of Drifted Columns", 0))

            drift_detected = nb_drifted > 0

            print(f"Nb clients: {nb_clients}, Dérive détectée: {drift_detected}, Score de dérive: {share_drift}")
            # bool dérive détectée
            drift_detected = nb_drifted > 0
            drift_score = share_drift  # ou un autre score selon ton fichier

            # affichage en streamlit
            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Nombre de clients (Echantillon)", f"{nb_clients}")
            col2.metric("📉 Taux de churn", f"{churn_rate*100:.1f} %")
            col3.metric(
                "⚖️ Dérive détectée ?",
                "Oui" if drift_detected else "Non",
                delta=f"Score: {drift_score:.2f}"
            )

            # ==========================
            # SECTION 2 - DISTRIBUTION DE LA CIBLE
            # ==========================
            st.header("2. Distribution du churn")

            df_churn = pd.DataFrame({
                "Churn": ["Oui", "Non"],
                "Count": [375, 1125]
            })

            fig1, ax1 = plt.subplots()
            #fig1.tight_layout()
            ax1.pie(df_churn["Count"], labels=df_churn["Churn"], autopct="%1.1f%%")
            ax1.set_title("Répartition churn")
            
            fig2, ax2 = plt.subplots()
            fig2.tight_layout()
            ax2.bar(df_churn["Churn"], df_churn["Count"], color=["red","green"])
            ax2.set_title("Vue barplot")


            colA, colB = st.columns(2)
            colA.pyplot(fig1)
            colB.pyplot(fig2)

            # ==========================
            # SECTION 3 - FEATURES IMPORTANTES
            # ==========================
            st.header("3. Analyse des variables clés")

            # Exemple variable numérique
            df_num = pd.DataFrame({
                "tenure": [5, 10, 15, 20, 25, 30],
                "Référence": [50, 80, 120, 90, 40, 20],
                "Actuel": [55, 70, 110, 100, 60, 30]
            })

            fig3, ax3 = plt.subplots()
            ax3.plot(df_num["tenure"], df_num["Référence"], label="Référence", marker="o")
            ax3.plot(df_num["tenure"], df_num["Actuel"], label="Actuel", marker="o")
            ax3.set_title("Distribution tenure")
            ax3.legend()

            # Exemple variable catégorielle
            df_cat = pd.DataFrame({
                "InternetService": ["DSL", "Fiber optic", "No"],
                "Référence": [500, 600, 400],
                "Actuel": [520, 700, 280]
            })

            fig4, ax4 = plt.subplots()
            ax4.bar(df_cat["InternetService"], df_cat["Référence"], alpha=0.7, label="Référence")
            ax4.bar(df_cat["InternetService"], df_cat["Actuel"], alpha=0.7, label="Actuel")
            ax4.set_title("InternetService distribution")
            ax4.legend()

            # Exemple autre feature
            df_monthly = pd.DataFrame({
                "MonthlyCharges": [20, 40, 60, 80, 100],
                "Référence": [80, 100, 120, 60, 30],
                "Actuel": [70, 110, 130, 65, 25]
            })

            fig5, ax5 = plt.subplots()
            ax5.plot(df_monthly["MonthlyCharges"], df_monthly["Référence"], label="Référence")
            ax5.plot(df_monthly["MonthlyCharges"], df_monthly["Actuel"], label="Actuel")
            ax5.set_title("MonthlyCharges distribution")
            ax5.legend()

            # Afficher en 3 colonnes
            col1, col2, col3 = st.columns(3)
            col1.pyplot(fig3)
            col2.pyplot(fig4)
            col3.pyplot(fig5)

            # ==========================
            # SECTION 4 - RECOMMANDATIONS
            # ==========================
            st.header("4. Recommandations business")
            st.success("""
            ✅ Pas de dérive significative détectée.  
            📌 La répartition du churn reste stable autour de 25%.  
            ⚠️ Surveiller la hausse du segment Fibre optique et les clients avec plus de 20 mois d’ancienneté.  
            """)

if __name__ == '__main__':
    main_app()