# Police Arrest Prediction — DNN & TabNet

Bu proje, polis tutuklama (arrest) verileri üzerinde bir tutuklamanın **"on-view arrest"** (olay yerinde/anlık tutuklama) olup olmadığını tahmin etmeyi amaçlayan bir ikili sınıflandırma çalışmasıdır. Model olarak bir **Deep Neural Network (DNN)** ve bir **TabNet Classifier** karşılaştırmalı olarak eğitilmiştir.

## İçindekiler

| Dosya | Açıklama |
|---|---|
| `dataset_linki.txt` | Ham veri setinin indirileceği link |
| `preprocessing.ipynb` | Veri temizleme ve ön işleme adımları |
| `DNNClassifier.ipynb` | PyTorch ile Deep Neural Network modeli |
| `TabNetClassifier.ipynb` | pytorch-tabnet ile TabNet modeli |
| `TheRookies_Sunum.pdf` | Proje sunumu |

## Proje Hakkında

Veri seti, gerçek polis tutuklama kayıtlarından oluşmaktadır ve her kayıt bir tutuklama olayına ait çeşitli özellikleri (konum, tarih, zaman gibi) içerir. Projenin amacı, bu özelliklerden yola çıkarak bir tutuklamanın polisin devriye sırasında olay yerinde anlık olarak mı ("on-view arrest"), yoksa önceden planlanmış/soruşturma sonucu mu gerçekleştiğini tahmin etmektir. Bu, kamu güvenliği verilerinin şeffaflığı ve polis operasyonlarının analiz edilmesi açısından anlamlı bir sınıflandırma problemidir.

Ham veri, `preprocessing.ipynb` içinde temizlenip modellenmeye hazır hale getirilir; ardından aynı işlenmiş veri seti üzerinde iki farklı derin öğrenme yaklaşımı — klasik bir DNN ve tablo verileri için özel olarak tasarlanmış TabNet mimarisi — eğitilip performansları karşılaştırılır.

## Yöntem

- **Hedef değişken**: `is_onview_arrest`
- **Ön işleme**: Kategorik değişkenler DNN için One-Hot Encoding, TabNet için Label Encoding ile sayısallaştırıldı. Düşük varyanslı özellikler `VarianceThreshold` ile elendi, DNN'de ayrıca `SelectKBest` (ANOVA F-testi) ile en anlamlı 1000 özellik seçildi.
- **Sınıf dengesizliği**: Eğitim setinde pozitif sınıf daha az örneklendiği için DNN'de `pos_weight` ile ağırlıklandırılmış `BCEWithLogitsLoss` kullanıldı.
- **Model seçimi**: DNN için rastgele arama (random search) ile hiperparametre optimizasyonu yapıldı (hidden_dim, dropout, learning rate).

## Sonuçlar

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| DNN Classifier | 0.795 | 0.570 | 0.607 | 0.588 | 0.833 |
| TabNet Classifier | 0.842 | 0.719 | 0.569 | 0.635 | 0.877 |

TabNet, daha yüksek doğruluk, precision ve ROC-AUC ile genel olarak daha güçlü performans göstermiştir; DNN ise recall açısından hafif üstündür.

## Kullanılan Teknolojiler

- **Python**
- **Pandas / NumPy** — veri işleme
- **Scikit-learn** — ön işleme, özellik seçimi, model değerlendirme
- **PyTorch** — DNN modelinin kurulumu ve eğitimi
- **pytorch-tabnet** — TabNet modelinin kurulumu ve eğitimi
- **Jupyter Notebook** — geliştirme ortamı
