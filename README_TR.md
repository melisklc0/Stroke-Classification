# Derin Öğrenme ve Bilgi Damıtımı (Knowledge Distillation) ile İnme (Stroke) Sınıflandırma Sistemi

**Proje Türü:** Lisans Bitirme Projesi & Modüler Veri Bilimi Portfolyosu  
📄 **[Bitirme Projesi Tez Raporu (PDF)](thesis/Stroke_Classification_Thesis_Report.pdf)**

## 🚀 Proje Özeti

Bu proje, bilgisayarlı tomografi (BT) taramalarından inme teşhisini hızlandırmak amacıyla tasarlanmış, yüksek düzeyde optimize edilmiş ve uçtan uca çalışan bir makine öğrenmesi boru hattı (pipeline) sunmaktadır. **Bilgi Damıtma (Knowledge Distillation - KD)** ve **Topluluk Öğrenme (Ensemble Learning)** yöntemlerini sentezleyerek, yüksek hesaplama gücü gerektiren karmaşık bir modelin bilgi birikimi, donanım dostu ve yüksek doğruluklu bir teşhis aracına başarıyla sıkıştırılmıştır.

Sistem, geleneksel tanı gecikmelerini ortadan kaldırarak hızlı bir **Klinik Karar Destek Sistemine** dönüşmekte; hafif mimarili "öğrenci" modellerin donanım (CPU/GPU) maliyetlerini ciddi oranda düşürürken, kendi karmaşık "öğretmen" modellerini performansta geride bırakabileceğini kanıtlamaktadır.

## 🛠 Tech Stack ve Metodolojiler

  * **Alan:** Bilgisayarlı Görü, Tıbbi Görüntüleme (BT Taramaları)
  * **Derin Öğrenme Mimarisi:** PyTorch, Transfer Öğrenme (ResNet, DenseNet, Inception, EfficientNet)
  * **İleri Teknikler:** Bilgi Damıtımı (Knowledge Distillation), Soft-Voting Topluluk Öğrenme (Ensemble), Veri Artırma (Data Augmentation)
  * **MLOps ve Altyapı:** `uv` (Hızlı Paket Yönetimi), YAML tabanlı konfigürasyon, Modüler Veri Boru Hatları

## 👥 Yazarlar

  * **Melis Kılıç**
  * **Esra Koç**

**Danışman:** Doç. Dr. Kali Gürkahraman

## 📊 Temel Başarılar ve Ölçülebilir Sonuçlar

  * **Öğretmeni Aşmak:** Temel (Baseline) Öğretmen model olan InceptionV3, %97.6 gibi güçlü bir F1 Skoruna ulaşmıştır. Bilgi Damıtma yöntemi sayesinde, hafif mimarili Öğrenci model (**EfficientNetB0**) çok daha düşük bir hesaplama maliyetiyle **%98.0 F1 Skoruna** ulaşarak kendi öğretmenini geride bırakmıştır.
  * **Maksimum Topluluk (Ensemble) Stabilitesi:** Sadece iki adet damıtılmış (KD) modelin (KD-EfficientNetB0 + B3) Soft-Voting stratejisiyle birleştirilmesi, genel hata oranını %1.7'ye düşürmüş ve **%98.2'lik** maksimum doğruluğa ulaşılmasını sağlamıştır.
  * **Güçlü Genellenebilirlik:** Model, tamamen harici bir Kaggle veri seti üzerinde doğrulanmıştır. Farklı tarayıcı kalibrasyonlarına ve renk matrislerine rağmen sistem, inme paternlerini gözden kaçırmama (yüksek recall) konusundaki kritik yeteneğini başarıyla korumuştur.

## 🔬 Mimari ve Veri Akışı (Data Pipeline)

### 1. Veri Mühendisliği ve Ön İşleme
* **Kaynaklar:** 
  * Ana Veri Seti: [T.C. Sağlık Bakanlığı Açık Veri Portalı](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)
  * Harici Test Seti (External Validation): [Kaggle Head CT Hemorrhage](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)
* **Sınıf Birleştirme:** Ham veri seti başlangıçta "İnme Yok", "Kanama (Bleeding)" ve "İskemi (Ischemia)" sınıflarını içermekteydi. Acil durum triyaj süreçlerini optimize etmek amacıyla "Kanama" ve "İskemi" sınıfları, ikili sınıflandırma (İnme Var / İnme Yok) hedefi doğrultusunda tek bir **"İnme (Stroke)"** sınıfı altında birleştirilmiştir.
* **Aşırı Öğrenmeyi (Overfitting) Önleme:** Sağlam bir öznitelik çıkarımı sağlamak için veri seti; döndürme ($\pm10^\circ$), yakınlaştırma ve yatay çevirme işlemleriyle 15.000 normalize tensör görüntüsüne çıkarılmıştır. Testler 3-Fold Çapraz Doğrulama (CV) stratejisiyle yürütülmüştür.

### 2. Eğitim Akışı (3 Faz)

1.  **Saf (Baseline) CNN Eğitimi:** 7 farklı CNN mimarisi özel bir sınıflandırma başlığı (256 nöronlu 2 Linear katman + Softmax) ile bağımsız olarak eğitilmiştir. InceptionV3 en uygun "Öğretmen" olarak belirlenmiştir.
2.  **Bilgi Damıtma (Knowledge Distillation):** InceptionV3'ün davranışları logit seviyesinde taklit edilerek, edindiği "derin bilgi" hafif öğrenci modellere aktarılmıştır.
3.  **Topluluk Ağı (Ensemble):** Tek model riskini ve önyargısını ortadan kaldırmak için, en başarılı KD modellerinin olasılıksal çıktıları Soft-Voting ile birleştirilmiştir.

## 💻 Kurulum ve Modüler Pipeline Kullanım Kılavuzu

Mimari, modüler ve tekrarlanabilir bir Veri Bilimi Boru Hattı olarak tasarlanmıştır. Tüm iş akışı `config.yaml` üzerinden yönetilmekte ve `run_pipeline.py` ile sıralı şekilde çalıştırılmaktadır.

### 1. Ortam Kurulumu

Projede bağımlılık yönetimi için ultra hızlı `uv` paket yöneticisi kullanılmaktadır.

#### UV Kurulumu (Eğer yüklü değilse)
*   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
*   **macOS / Linux (Bash):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*(Not: Kurulumdan sonra terminali kapatıp yeniden açmanız gerekebilir.)*

#### Sanal Ortam ve Paketlerin Kurulumu
```bash
uv venv
uv pip install .
```

### 2. Veri Seti Düzenleme Konfigürasyonları

`--step data` komutu yerel veri yapınıza dinamik olarak uyum sağlar. Ham verinizin durumuna uygun olan senaryoyu seçebilirsiniz:

  * **Senaryo A: Ham Görselleriniz Varsa (Raw Images)**  
  Elinizde sadece ham resimler bulunuyorsa, `dataset/` klasörü yapınızı şu şekilde düzenleyin:

    ```text
    dataset/
    ├── Stroke/       (Tüm hasta görüntüleri)
    └── No-Stroke/    (Tüm sağlıklı görüntüler)
    ```

    Bu durumdayken `uv run python run_pipeline.py --step data` komutunu çalıştırdığınızda, sistem bunları otomatik olarak işler, artırır (augmentation) ve `dataset/Fold1`, `dataset/Fold2` vb. şeklinde ayırır.

  * **Senaryo B: Fold'larınız Zaten Hazırsa (Pre-split)**  
  Eğer veriniz daha önceden ayrılmışsa, klasörleri doğrudan `dataset/` dizini altına yerleştirin:

    ```text
    dataset/
    ├── Fold1/
    ├── Fold2/
    └── Fold3/
    ```

    Bu durumda **data adımını çalıştırmanıza gerek yoktur.**  
    Sistem bu hiyerarşiyi otomatik algılayacak ve doğrudan eğitim aşamasına (`--step train_cnn`) hazır olacaktır.

### 3. Pipeline Akışının Çalıştırılması (`run_pipeline.py`)

**Adım 1: Kümeleme, Dengeleme ve Artırma (Augmentation)**  
Veri setini `config.yaml` ayarlarınıza (örn. K-Fold adedi = 3) göre hazırlayın.

```bash
uv run python run_pipeline.py --step data
```

**Adım 2: Saf (Baseline) CNN Eğitimi (Öğretmen)**  
`config.yaml` üzerinden bir model seçin (örn. `inceptionv3`) ve temel performansı (baseline) belirlemek için eğitin. *(Not: Inception kullanıyorsanız konfigürasyonda `image_size: 299` ayarlandığından emin olun).*

```bash
uv run python run_pipeline.py --step train_cnn
```

**Adım 3: Knowledge Distillation (KD) Eğitimi (Öğrenci)**  
Eğitilmiş InceptionV3 `.pth` ağırlığının dosya yolunu gösterin ve yeni bir öğrenci model (örn. `efficientnetb0`) eğitin. Hata fonksiyonu hiperparametreleri kod içerisinde $\alpha=0.7$ ve $\tau=5.0$ olarak optimize edilmiştir.

```bash
uv run python run_pipeline.py --step train_kd
```

**Adım 4: Topluluk Testleri (Soft-Voting Ensemble)**  
En yüksek performanslı KD Öğrenci modellerinizin `.pth` yollarını `config.yaml` dosyasındaki `ensemble -> weights` bölümünün altına ekleyin. `validation_mode` (internal/external) seçiminizi yapıp süreci başlatın:

```bash
uv run python run_pipeline.py --step ensemble
```

> **Görsel Çıktılar ve Kayıt Yönetimi:**
>
>   - **Metrikler ve Grafikler:** `results/cnn/`, `results/kd/` veya `results/ensemble/` klasörlerine otomatik olarak kaydedilir. Bu çıktılar; Confusion Matrix (Karmaşıklık Matrisi/Isı Haritası), Eğitim Kayıp (Loss) eğrileri ve (F1, Recall vb.) tüm sınıflandırma raporlarının PNG ve TXT formatlarını içerir.
>   - **Orijinal Tez Sonuçları:** `thesis/results/` dizininde arşivlenmiştir.
>   - **Model Ağırlıkları (.pth):** Kontrol noktaları (checkpoints) güvenli bir şekilde `checkpoints/` klasörüne kaydedilir, ancak deponun boyutunu şişirmemek adına `.gitignore` dosyasına eklenmiştir.