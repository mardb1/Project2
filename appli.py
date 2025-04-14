import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from methode_h import classification_ward
from methode_k import kmeans

def check_password():
    """Returns True if the user entered the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "dbib.mariam":
            st.session_state["password_correct"] = True
            st.session_state["password"] = ""  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Please enter the password to access the application")
        st.stop()
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        st.stop()

# Check password before showing the app
check_password()

# Background setup with text styling
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        
        /* Text and title styling */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stRadio > label, .stButton > button, .stSelectbox > label {{
            color: black !important;
            text-shadow: 
                -1px -1px 0 white,
                1px -1px 0 white,
                -1px 1px 0 white,
                1px 1px 0 white !important;
        }}
        
        /* Widget styling */
        .stTextInput, .stSlider, .stDataFrame, .stButton > button, .stFileUploader {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ddd !important;
        }}
        
        /* Special styling for main title */
        .stApp h1 {{
            font-size: 2.5rem !important;
            text-align: center;
            padding: 15px;
            background-color: rgba(255,255,255,0.7) !important;
            border-radius: 10px;
            margin-bottom: 30px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def afficher_dendrogramme(donnees, labels):
    Z = linkage(donnees, 'ward')
    fig, ax = plt.subplots(figsize=(6, 3))
    dendrogram(Z, labels=labels, orientation='right', ax=ax)
    plt.title("Dendrogramme", color='black', pad=20)
    st.pyplot(fig)

def afficher_historique(historique, noms_etudiants):
    st.markdown("## Historique des Regroupements")
    
    groupes = {i: (i,) for i in range(len(noms_etudiants))}
    
    for step in historique:
        cluster_a, cluster_b = map(tuple, step['fusion'])
        
        indices_a = groupes[cluster_a] if cluster_a in groupes else cluster_a
        indices_b = groupes[cluster_b] if cluster_b in groupes else cluster_b
        
        noms_a = [noms_etudiants[i] for i in indices_a]
        noms_b = [noms_etudiants[i] for i in indices_b]
        
        st.markdown(f"**Étape {step['step']}** (Distance euclidienne: {step['distance']:.2f}):")
        st.markdown(f"{', '.join(noms_a)} + {', '.join(noms_b)}")
        st.markdown("---")
        
        nouveau_groupe = indices_a + indices_b
        nouveau_id = nouveau_groupe
        
        for id in [cluster_a, cluster_b]:
            if id in groupes:
                del groupes[id]
        
        groupes[nouveau_id] = nouveau_groupe

def afficher_kmeans(resultats, noms_etudiants):
    st.markdown("## Résultats K-means")
    
    couleurs = ['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0']
    fig, ax = plt.subplots(figsize=(10, 7))

    # Points étudiants (triangles)
    for cluster_id in range(len(resultats['centroides'])):
        indices = np.where(resultats['labels'] == cluster_id)[0]
        ax.scatter(
            donnees[indices, 0], donnees[indices, 1],
            marker='^', c=couleurs[cluster_id], s=80,
            edgecolor='k', alpha=0.8, label='Étudiant' if cluster_id == 0 else ""
        )

    # Centroïdes (carrés rouges)
    for cluster_id, centroid in enumerate(resultats['centroides']):
        ax.scatter(
            centroid[0], centroid[1], marker='s',
            s=150, c='red', edgecolor='k', linewidth=1.5,
            label='Centroïde' if cluster_id == 0 else ""
        )

    # Superpositions
    for cluster_id in range(len(resultats['centroides'])):
        indices = np.where(resultats['labels'] == cluster_id)[0]
        for i in indices:
            if np.linalg.norm(donnees[i] - resultats['centroides'][cluster_id]) < 0.1:
                ax.scatter(
                    donnees[i, 0], donnees[i, 1],
                    marker='^', c=couleurs[cluster_id], s=120,
                    edgecolor='k', linewidth=0.5, zorder=5
                )

    plt.title("Visualisation des Clusters", color='black', pad=20)
    plt.xlabel("Note 1", color='black')
    plt.ylabel("Note 2", color='black')
    
    handles = [
        plt.Line2D([], [], marker='s', color='red', markersize=10, label='Centroïde', linestyle='None'),
        plt.Line2D([], [], marker='^', color='blue', markersize=10, label='Étudiant', linestyle='None')
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)

def main():
    # Set your background image URL here
    set_background("https://raw.githubusercontent.com/mardb1/finalbg/main/bg_final2.jpg.jpg")
    
    global donnees
    
    # Main title and description
    st.title("Classification hiérarchique - Classification par K-means")
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;'>
    <h3 style='color:black;'>Welcome!</h3>
    <p style='color:black;'>Veuiller importez un fichier Excel contenant les données.</p>
    </div>
    """, unsafe_allow_html=True)
    
    fichier = st.file_uploader(
        "📤 Importer un fichier Excel", 
        type=["xlsx", "xls"],
        help="Importer un fichier Excel avec les notes des étudiants"
    )
    
    if fichier:
        try:
            df = pd.read_excel(fichier, index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.empty:
                st.error("Aucune donnée valide après nettoyage")
                return
            
            noms_etudiants = df.index.tolist()
            donnees = df.values
            
            st.success(f"✅ {len(df)} éléments chargés")
            st.dataframe(df)
            
            methode = st.radio("Méthode de classification :", ["Hiérarchique", "K-means"])
            
            if methode == "Hiérarchique":
                if st.button("Lancer l'Analyse Hiérarchique"):
                    with st.spinner("Calcul en cours..."):
                        historique = classification_ward(donnees)
                        st.success("Classification hiérarchique terminée !")
                        
                        tab1, tab2 = st.tabs(["Dendrogramme", "Historique"])
                        with tab1:
                            afficher_dendrogramme(donnees, noms_etudiants)
                        with tab2:
                            afficher_historique(historique, noms_etudiants)
            
            elif methode == "K-means":
                k = st.selectbox(
                    "Nombre de clusters :",
                    options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                    index=1
                )
                
                if st.button("Lancer l'Analyse K-means"):
                    with st.spinner("Calcul en cours..."):
                        resultats = kmeans(donnees, k=k)
                        st.success("Clustering K-means terminé !")
                        afficher_kmeans(resultats, noms_etudiants)
                        
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()