# SignFlow — Pengenalan Bahasa Isyarat ASL Real-Time

**SignFlow** adalah sistem penerjemah bahasa isyarat **American Sign Language (ASL)** secara real-time berbasis **Computer Vision** dan **Deep Learning**. Sistem ini mengubah gestur tangan dan tubuh bagian atas dari kamera webcam menjadi teks secara instan.

## Arsitektur Sistem
- **Ekstraksi Landmark:** MediaPipe Holistic melacak 48 titik kunci spasial (12 pose tubuh bagian atas, 42 tangan kiri, 42 tangan kanan) + 2 mask deteksi tangan, menghasilkan vektor 98 dimensi per frame yang dinormalisasi terhadap pusat dan lebar bahu.
- **Pemodelan Sekuens:** 2-layer Bidirectional GRU (`SignBiGRU`, 256 unit per arah) dengan *Temporal Average Pooling* pada *sliding window* 32 frame.
- **Hasil Evaluasi Benchmark (Test Set / Unseen Signers):**
  - **Akurasi Top-1:** 75,7%
  - **Akurasi Top-3:** 89,1%
  - **Kosakata Target:** 65 kata ASL + 1 kelas `idle`.

## Struktur Proyek
```text
SignFlow-AI/
├── models/
│   ├── model.pt
│   └── label_map.json
├── src/
│   ├── __init__.py
│   ├── model_arch.py
│   ├── feature_extractor.py
│   └── utils.py
├── app_webcam.py
├── requirements.txt
└── README.md
```

## Instalasi & Cara Menjalankan
1. Install dependensi yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
2. Pastikan file `model.pt` dan `label_map.json` berada di dalam folder `models/`.
3. Jalankan inferensi webcam secara real-time:
   ```bash
   python app_webcam.py
   ```
   *(Gunakan `python app_webcam.py --camera 1` jika memakai webcam eksternal.)*

### Kontrol Keyboard
- **`a`**: Ganti mode rasio aspek (`3:4` / `9:16` / `Raw`)
- **`c`**: Reset / bersihkan buffer 32 frame
- **`q`**: Keluar dari aplikasi
