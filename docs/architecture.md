# Mimari ve genişletme noktaları

## Tek ortam, iki çalışma biçimi

`World` ekran, saat, dosya veya öğrenme sistemi bilmez. Zaman yalnızca tam sayı
`frame` sayacıyla ilerler. Bu sayede gün sonları kayan nokta birikimine bağlı değildir.

Eğitim `World.step([ada_action, atlas_action])` çağırır. Bu metot `begin_actions`,
24 kez `tick` ve `finish_actions` uygular. Gözlemevi de aynı metotları kullanır,
ancak 24 fizik karesini gerçek zamana dağıtır. Parite testi iki yolun gözlemlerinin,
ödüllerinin ve böcek durumlarının birebir aynı kaldığını doğrular.

`finish_actions`, pasif anten ölçümlerini günceller; seçilen eylem `SNIFF` ise daha
uzun menzilden örnek alır. Ağ yalnızca bu ölçümleri ve yerel duyuları görür.
Motor eylemi başladıktan sonra bir karar aralığı boyunca değişmez.

## Ajanlar

Her böcek için bağımsız `Brain` bulunur. Replay kayıtları `(s, a, r, s_next, done)`
biçimindedir. İki ajan aynı dünyanın etkilerini hisseder, ancak birbirlerinin
öğrenme ağırlıklarına erişmez. Bu, küçük bir bağımsız çok ajanlı RL başlangıcıdır;
merkezi eleştirmen veya popülasyon/genetik optimizasyonu değildir.

Double DQN hedefi:

```text
chosen = argmax Q_online(next_state)
target = reward + gamma * (1 - done) * Q_target(next_state, chosen)
```

Hedef ağ 200 optimizer güncellemesinde bir kopyalanır. Replay 30.000 karar tutar;
256 deneyden sonra her dört kararda bir 64 örnekle öğrenilir. `gamma=0.985` bir
0,8 saniyelik karar için tanımlıdır. Karar süresi değişirse etkili zaman ufku da değişir.

Çiftleşmede kısa mesafeli abdomen yöneltme refleksi, `COURT` motor eyleminin parçasıdır.
Ağ bu eylemin **ne zaman** seçileceğini öğrenir; refleks yiyecek aramaz veya akvaryumda
uzak eşe yol planlamaz. Başlangıç politikası bu eylemi de rastgele deneyebilir.

## Deneylerin ayrılması

1. Eğitim dünyaları: `training_seed + episode_index`.
2. En iyi modeli seçen doğrulama dünyaları: `training_seed + 1_000_000 … + 1_000_004`.
3. Son karşılaştırma dünyaları: kullanıcı tarafından ayrı seçilir, varsayılan `10000 …`.

Kullanıcı aynı tohumları özellikle seçerse ayrılık bozulabilir. Örnek modelin eğitim,
doğrulama ve test tohumları ayrı tutulur. Doğrulama aynı beş dünya üzerinde tekrarlandığı
için en iyi modeli bu örneklere fazla uydurmak mümkündür; son test bunun için ayrıdır.

Gözlemevi modelleri `training=False` ile açar: replay belleği oluşturulmaz ve
`remember` çağrısı hata verir. Kayıtlar yerinde değiştirilmez. `L` ile yüklenen
model yeni bir gözlem bölümü başlatır. Eğitim ve gözlem ayrı işlemlerde aynı anda
çalışabilir; atomik dosya değişimi yarım checkpoint okumayı engeller.

## Sonraki deneyler

- **Su:** `Food` benzeri `Water` kaynakları, ayrı koku kanalı, susuzluk durumu ve ödül.
  Mevcut besin gözlemlerinin anlamını sessizce değiştirmeyin.
- **Düşman:** yeni aktör tipi ve tehlike duyuları; önce sabit davranışlı rakip, sonra
  öğrenen rakip eklemek öğrenme problemini aşamalı büyütür.
- **Üreme müfredatı:** beslenmiş, yakın başlayan çiftlerle kur eylemini eğitmek ve
  giderek normal ortama geçmek, çok seyrek çiftleşme ödülünü öğrenmeyi kolaylaştırır.
  Böyle bir eğitim dağılımı raporlarda açıkça belirtilmelidir.
- **Yavrular:** yumurtadan çıkışı açmadan önce ajan kimliği, yeni beyin başlatılması,
  nüfus sınırı ve besin bütçesi tanımlanmalıdır. Şimdiki iki ajan varsayımı bilinçlidir.
- **Uzun süreli bellek:** anten geçmişi için GRU veya birkaç gözlemi yığma eklenebilir.
  Mevcut ağ yalnızca son ölçüm, ölçüm farkı ve son eylemi hatırlar.

Yeni gözlem/eylem eklenirse `OBS_NAMES`, checkpoint sürümü ve ilgili testler birlikte
güncellenmelidir. Eski bir model farklı duyularla sessizce çalıştırılmamalıdır.
