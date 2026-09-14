# Flowers102 EfficientNet Classifier

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arielshakaramiro/flowers102-efficientnet-classifier/blob/main/notebook/flowers102_efficientnet_classifier.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Model klasifikasi gambar dengan pendekatan *transfer learning* untuk dataset [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (102 jenis bunga), dibangun dengan **PyTorch** dan backbone **EfficientNet-B1** yang sudah pretrained di ImageNet. Repo ini berisi notebook training/evaluasi lengkap dan script serving **FastAPI** untuk inference real-time.

## Hasil

Seluruh angka di bawah berasal dari eksekusi nyata notebook training di GPU (Google Colab), bukan estimasi. Rincian lengkap per kelas ada di dalam notebook.

| Metrik | Nilai |
|---|---|
| Akurasi validasi terbaik | 89,41% (epoch 8/15) |
| Akurasi test (6.149 gambar yang tidak pernah dilihat model) | 87,95% |
| Precision test (weighted) | 89,76% |
| Recall test (weighted) | 87,95% |
| Parameter yang dilatih | 130.662 (hanya bagian classifier) |
| Total parameter | 6.643.846 |

![Kurva training dan validasi](docs/training_curves.png)

Akurasi validasi memuncak sekitar epoch 8, lalu sedikit menurun setelahnya — indikasi overfitting ringan, wajar mengingat data training resminya cukup kecil (1.020 gambar untuk 102 kelas). Karena itu, checkpoint terbaik (`best.pt`) yang dipakai untuk evaluasi dan serving, bukan checkpoint terakhir.

Performa antar-kelas tidak merata: beberapa kelas mencapai precision/recall sempurna 1.00 (mis. *bird of paradise*, *black-eyed susan*), sementara beberapa kelas yang lebih sulit/mirip secara visual jauh lebih rendah (mis. *mallow* precision 0,41, *japanese anemone* recall 0,46). Rata-rata weighted di atas menyembunyikan variasi ini — lihat `docs/results.json` dan classification report lengkap di notebook untuk gambaran menyeluruh.

![Contoh gambar validasi dengan label kelas](docs/sample_images.png)

## Dataset

M-E. Nilsback dan A. Zisserman, *"Automated Flower Classification over a Large Number of Classes"*, Indian Conference on Computer Vision, Graphics and Image Processing, 2008. Notebook mengunduh dataset langsung dari sumber resminya dan memakai split train/val/test resmi dari paper aslinya (`setid.mat`) supaya hasilnya reproducible.

Mapping nama kelas (index → nama bunga) diambil saat runtime dari referensi publik yang sudah dipakai luas (`cat_to_name.json`), bukan diketik manual — setelah draf awal yang ditulis manual ternyata ketahuan berisi 104 entri, bukan 102, saat proses verifikasi.

## Struktur proyek

```
.
├── notebook/
│   └── flowers102_efficientnet_classifier.ipynb   # training + evaluasi, terverifikasi di GPU
├── serving/
│   └── main.py                                    # endpoint inference FastAPI
├── docs/
│   ├── training_curves.png
│   ├── sample_images.png
│   └── results.json
├── requirements.txt
└── LICENSE
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Training

Buka `notebook/flowers102_efficientnet_classifier.ipynb` di Google Colab (disarankan pakai GPU runtime), lalu jalankan semua sel. Notebook akan:

1. Mengunduh dataset Oxford Flowers 102 beserta split resminya.
2. Mengambil mapping nama kelas.
3. Melatih bagian classifier dari EfficientNet-B1 pretrained.
4. Mencatat loss/accuracy/precision/recall tiap epoch, menyimpan `checkpoints/best.pt` dan `checkpoints/last.pt`.
5. Mengevaluasi di test set dan mencetak classification report lengkap.

File bobot model masuk `.gitignore` — setelah training, `checkpoints/best.pt` akan ada di lokal tapi tidak ikut ter-commit ke repo ini.

## Serving

```bash
uvicorn serving.main:app --reload
```

```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@path_ke_gambar.jpg"
```

Respons:

```json
{"predicted_class": "sunflower", "confidence": 0.93}
```

Script serving memakai pipeline preprocessing dan arsitektur model yang persis sama dengan saat training — ketidakcocokan di salah satunya adalah penyebab paling umum prediksi yang salah tanpa error di proyek transfer learning seperti ini.

## Lisensi

MIT — lihat [LICENSE](LICENSE).
