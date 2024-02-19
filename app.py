import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def run():
    st.title("Meme Kanseri Veriseti ile Model Eğitimi")
    uploaded_file = st.sidebar.file_uploader("Lütfen Verisetinizi Yükleyiniz.", type=["csv"])
    df = pd.read_csv(uploaded_file)
    set_clf = st.sidebar.selectbox("Select Model", ("KNN", "SVM", "Naive Bayes"))

    #df = pd.read_csv("data.csv")
    st.write("Verisetinin İlk 10 Satırı")
    st.write(df.head(10))
    st.write("Verisetindeki Nitelikler")
    st.write(list(df.columns))

    # Gorev 2
    df.dropna(axis=1, inplace=True)
    df.drop("id", axis=1, inplace=True)
    st.write("Verisetinin Son 10 Satırı")
    st.write(df.tail(10))
    M = df[df["diagnosis"] == "M"]
    B = df[df["diagnosis"] == "B"]

    # M ve B degerlerinin  0 ve 1'e donusumu
    df["diagnosis"] = [1 if values == "M" else 0 for values in df["diagnosis"]]

    # X ve Y verilerimiz:
    y = df["diagnosis"].values
    X = df.drop("diagnosis", axis=1)

    # Korelasyon Matrisi
    st.write("Korelasyon Matrisi")
    norm_df = df - df.mean() / df.std()
    matrix = np.triu(norm_df.corr())
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(16,10))
    sb.heatmap(norm_df.corr(), mask=matrix)
    st.write(fig)

    #Plot
    st.write("Yarıçap ve Doku Verileriyle Meme Kanserinin Dağılım Grafiği")
    fig = plt.figure(figsize=(16, 12))
    plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Kotu", alpha=0.7)
    plt.scatter(B.radius_mean, B.texture_mean, color="blue", label="Iyi", alpha=0.7)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.legend()
    st.pyplot(fig)

    params_grid_knn = [{"n_neighbors":[1,2,3,4,5,6,7,8,9,10]}]
    params_grid_svm = [{"C":[1,2,3,4,5,6,7,8,9,10],"degree":[1,2,3,4,5]}]

    #Seçilen Classifier için GridSearch işlemi
    if set_clf == "KNN":
        clf = GridSearchCV(KNeighborsClassifier(), params_grid_knn)
    elif set_clf == "SVM":
        clf = GridSearchCV(SVC(), params_grid_svm)
    else:
        clf = GaussianNB()

    # Verinin train ve test olarak ayrılması
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    st.write(f"Classifier = {set_clf}")
    st.write(f"Accuracy =", accuracy)
    st.write(f"Precision =", precision)
    st.write(f"Recall =", recall)
    st.write(f"F1 Score = ", f1)
    if (set_clf == "KNN") or (set_clf == "SVM"):
        st.write("En iyi parametre değerleri:")
        st.write(clf.best_params_)

    # Confusion matrix'i görsel olarak göster
    fig, ax = plt.subplots()
    sb.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Tahmin Edilen Etiketler")
    plt.ylabel('Gerçek Etiketler')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    st.write("")
    st.write("Dikkat etmemiz gereken metrikler her verisetine göre değişiklik gösterecektir.\n"
             "Kanser gibi bir durumu incelerken bakmamız gereken en önemli metrik ise FN değerleri olmalıdır.\n"
             "FN değeri başka bir deyişle Recall, kanser olduğu halde modelimizin sağlıklı olduğunu söylediği değerlerdir.")