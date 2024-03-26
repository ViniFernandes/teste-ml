import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier

file_name = 'files/mini_gm_public_v0.1 (1).p'

# Lendo o arquivo pickle
obj = pickle.load(open(file_name, "rb"))
df = pd.DataFrame(columns=['syndrome_id', 'subject_id', 'image_id', 'img_encoding'])

# Montando o Dict como um DataFrame
for syndrome_id in obj:
    for subject_id,x in obj[syndrome_id].items():
        for image_id, img_encoding in x.items():
            new_row = pd.Series({'syndrome_id': syndrome_id, 'subject_id': subject_id, 'image_id': image_id, 'img_encoding': img_encoding})
            df = pd.concat([df, pd.DataFrame([new_row], columns=new_row.index)]).reset_index(drop=True)

# Separando os dados dos labels
X = np.array(df['img_encoding'].tolist())
Y = np.array(df['syndrome_id'].tolist())

# Separando a base de treino e 20% para a base de teste
treino_x, teste_x, treino_y, teste_y = train_test_split(X, Y, test_size=0.2, random_state=10)

# Realizando a redução dos atributos utlizando TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=10)
z = tsne.fit_transform(treino_x)
df_tsne = pd.DataFrame()
df_tsne["y"] = treino_y
df_tsne["componente-1"] = z[:,0]
df_tsne["componente-2"] = z[:,1]

# Gerando o Gráfico e salvando em um arquivo
sns.set(rc = {'figure.figsize':(12,12)})
sns.scatterplot(x="componente-1", y="componente-2", hue=df_tsne.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df_tsne).set(title="T-SNE")
plt.savefig('tsne_projection.png')

# Definir o número de folds para a validação cruzada
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Inicializar um dataframe para armazenar os resultados da classificação
results_df = pd.DataFrame(columns=['Fold', 'Cosine AUC', 'Euclidean AUC'])

# Inicializar listas para armazenar as predições
cosine_predictions = []
euclidean_predictions = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    # Obter os conjuntos de treino e teste para o fold especifico
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]

    # Separar os dados de treino
    X_train = np.array(train_data['img_encoding'].tolist())
    y_train = np.array(train_data['syndrome_id'])

    # Separar os dados de teste
    X_test = np.array(test_data['img_encoding'].tolist())
    y_test = np.array(test_data['syndrome_id'])

    # Calcular a distância de cosseno e a distância euclidiana
    cosine_distances = np.dot(X_test, X_train.T)
    euclidean_distances = np.linalg.norm(X_test[:, np.newaxis] - X_train, axis=2)

    # Classificar cada imagem usando KNN para distância de cosseno
    knn_cosine = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_cosine.fit(X_train, y_train)
    cosine_pred = knn_cosine.predict_proba(X_test)
    cosine_predictions.extend(cosine_pred)

    # Classificar cada imagem usando KNN para distância euclidiana
    knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_euclidean.fit(X_train, y_train)
    euclidean_pred = knn_euclidean.predict_proba(X_test)

    euclidean_predictions.extend(euclidean_pred)

    # Calcular a AUC (Area Under the ROC Curve) para este fold para ambas as distâncias
    cosine_auc = roc_auc_score(y_test, cosine_pred, average='macro', multi_class='ovr')
    euclidean_auc = roc_auc_score(y_test, euclidean_pred, average='macro', multi_class='ovr')


    # Armazenar os resultados da AUC no dataframe
    new_row = pd.DataFrame([{'Fold': fold + 1, 'Cosine AUC': cosine_auc, 'Euclidean AUC': euclidean_auc}])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Salvar os resultados da AUC em um arquivo CSV
results_df.to_csv('evaluation_results.csv', index=False)

true_labels = df['syndrome_id'].tolist()

# Binarizando os labels verdadeiros para cada classe
binarized_labels = label_binarize(true_labels, classes=np.unique(true_labels))


# Calcular as curvas ROC e as áreas sob a curva (AUC) para cada classe
fpr_cosine = {}
tpr_cosiine = {}
roc_auc_cosine = {}

fpr_euclid = {}
tpr_euclid = {}
roc_auc_euclid = {}
# Para cada classe
for i in range(10):
    fpr_cosine[i], tpr_cosiine[i], _ = roc_curve(binarized_labels[:, i], np.array(cosine_predictions)[:, i])
    roc_auc_cosine[i] = auc(fpr_cosine[i], tpr_cosiine[i])

    fpr_euclid[i], tpr_euclid[i], _ = roc_curve(binarized_labels[:, i], np.array(euclidean_predictions)[:, i])
    roc_auc_euclid[i] = auc(fpr_euclid[i], tpr_euclid[i])

# Calcular a média geral da AUC para a distancia de cosseno
all_fpr_coss = np.unique(np.concatenate([fpr_cosine[i] for i in range(10)]))
mean_tpr_coss = np.zeros_like(all_fpr_coss)
for i in range(10):
    mean_tpr_coss += np.interp(all_fpr_coss, fpr_cosine[i], tpr_cosiine[i])

mean_tpr_coss /= 10
macro_auc = auc(all_fpr_coss, mean_tpr_coss)

# Calcular a média geral da AUC para a distancia euclidiana
all_fpr_euclid = np.unique(np.concatenate([fpr_euclid[i] for i in range(10)]))
mean_tpr_euclid = np.zeros_like(all_fpr_euclid)
for i in range(10):
    mean_tpr_euclid += np.interp(all_fpr_euclid, fpr_euclid[i], tpr_euclid[i])

mean_tpr_euclid /= 10
macro_auc_euclid = auc(all_fpr_euclid, mean_tpr_euclid)


# Gerar o gráfico com as duas curvas roc
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.plot(all_fpr_coss, mean_tpr_coss, color='black', linestyle='-', lw=2,
         label='ROC Cosseno(AUC = {0:.2f})'.format(macro_auc))
plt.plot(all_fpr_euclid, mean_tpr_euclid, color='blue', linestyle='-', lw=2,
         label='ROC Euclidiana(AUC = {0:.2f})'.format(macro_auc))

plt.xlabel('Falso positivo (FPR)')
plt.ylabel('Verdadeiro positivo (TPR)')
plt.title('Curva ROC')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('curva_roc.png')