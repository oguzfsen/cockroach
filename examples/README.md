# Örnek besin arama modeli

**Arşiv: aşağıdaki eğitim ve ölçümler 0.1.0 ortamına aittir** (sabit besin, 6 gün,
günde bir öğün). 0.2 sürümü bu modeli ağırlıklarını koruyarak yeni 30 günlük,
hareketli ve rekabetçi besin ortamına aktarır. Eski dosya değiştirilmez. Bu rapor
yeni kuralların başarısını ölçmez; aşağıdaki değerlendirme komutu güncel ortamda
yeni bir rapor üretir. Yeni sürüm hazırlanırken yeniden eğitim yapılmamıştır.

`forager.npz`, Ada ve Atlas için ayrı ağları içerir. Görüntüleyici yalnızca çıkarım yapar.
Model gerçek pekiştirmeli öğrenmeyle üretildi; önceden yazılmış besin arama politikası
veya uzman gösterimi kullanılmadı.

```bash
cockroach view --model examples/forager.npz
cockroach evaluate --model examples/forager.npz --episodes 20 --seed 10000
```

## Eğitim koşulları

```bash
cockroach train --steps 120000 --seed 42 --output runs/reference.npz
```

- 120.000 ortak karar, her böcek için 120.000 ayrı deney; 266 tamamlanan bölüm.
- Varsayılan 960×640 arena, 30 Hz fizik, 0,8 saniye karar ve 60 saniye/gün.
- Yaklaşık 1.600 simülasyon günü; bu bilgisayarda doğrulamalar dahil 102,9 saniye.
  Hız donanıma ve Python/NumPy sürümüne bağlıdır.
- Python 3.10.12, NumPy 2.2.6. Görüntüleme kontrolü pygame-ce 2.5.8 ile yapıldı.
- Model seçimi beş ayrı dünyada: `1000042 … 1000046`.
  En iyi aday bu koşuda son karar olan 120.000'de seçildi.
- [training.csv](training.csv) bölüm kayıtlarını,
  [validation.jsonl](validation.jsonl) tüm aday doğrulamalarını içerir.

## Ayrı test

`10000 … 10019` tohumlarında 20 eşleştirilmiş bölüm, toplam 120 gün ve 240 besin.
Bu tohumlar eğitimde veya model seçiminde kullanılmadı. Değerlendirmede keşif kapalı.

| Ölçüm | Rastgele | Öğrenilmiş |
| --- | ---: | ---: |
| Toplanan besin | 48 / 240 | 193 / 240 |
| Besin oranı | %20,0 | **%80,4** |
| Bölüm başına ortalama skor | −33,57 | +198,59 |
| Tamamlanan çiftleşme | 0 | 1 |

Bölüm başına besin farkı **+7,25**; 2.000 eşleştirilmiş bootstrap örneklemesinde %95
aralık **[+6,60; +7,85]**. Ham bölüm sonuçları: [evaluation.json](evaluation.json).
Bu tek eğitim tohumu için bir ölçümdür; farklı eğitimlerin aynı sonucu vermesi beklenmez.
Rastgele başlangıç, yakın zamanda karşılaşma veya farklı koşullar başarıyı etkileyebilir.

Bu örnek ağırlıklı olarak besin arama davranışı içindir; bir çiftleşme, üreme davranışının
sağlam biçimde öğrenildiğini göstermez. Çiftleşme mekaniği `F2` ile ayrıca incelenebilir;
bu kontrollü sahne öğrenilmiş çiftleşme başarısı sayılmaz.
