# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Charger le modèle KMeans depuis le fichier
with open('/Users/apple/Desktop/Model-Kmeans.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Charger les données d'exemple (vous devrez remplacer cela par vos propres données)
# Assurez-vous que les données ont des caractéristiques similaires à celles que vous avez utilisées pour entraîner le modèle.
# Par exemple, montant total des achats, fréquence d'achat, date du dernier achat, etc.
example_data = pd.DataFrame({
    'CustomerID': [1, 2, 3],
    'TotalAmount': [500, 1000, 200],
    'Frequency': [5, 10, 2],
    'LastPurchaseDate': ['2023-01-15', '2023-01-10', '2023-01-20']
})

# Convertir la colonne 'LastPurchaseDate' en datetime
example_data['LastPurchaseDate'] = pd.to_datetime(example_data['LastPurchaseDate'])

def calculate_days_since_last_purchase(last_purchase_date):
    current_date = datetime.now()
    days_since_last_purchase = (current_date - last_purchase_date).days
    return days_since_last_purchase

def main():
    st.title("Clustering des Clients pour une Boutique en Ligne")

    # Afficher les données d'exemple
    st.subheader("Exemple de Données Client:")
    st.write(example_data)

    # Interface utilisateur pour entrer de nouvelles données
    st.sidebar.header("Entrer de Nouvelles Données Client")
    total_amount = st.sidebar.number_input("Montant Total des Achats", value=0)
    frequency = st.sidebar.number_input("Fréquence d'Achat", value=0)
    last_purchase_date_str = st.sidebar.date_input("Date du Dernier Achat", datetime.now())

    # Convertir la date du dernier achat en datetime
    last_purchase_date = pd.to_datetime(last_purchase_date_str)

    # Calculer la différence en jours depuis le dernier achat
    days_since_last_purchase = calculate_days_since_last_purchase(last_purchase_date)

    # Prédiction avec le modèle KMeans
    new_data = np.array([[total_amount, frequency, days_since_last_purchase]])
    cluster_prediction = kmeans_model.predict(new_data)

    # Afficher la prédiction
    st.write(f"Le nouveau client est prédit appartenir au cluster : {cluster_prediction[0]}")

if __name__ == "__main__":
    main()
