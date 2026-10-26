# Proje notları

Bu proje ROS kullanmaz; Python ile bağımsız bir simülasyondur.
ROS eklenirse bu bilgisayar yalnızca kod yazmak için kullanılır. Gerçek ROS ortamı
burada değildir; kullanıcı kodu rsync ile hedefe gönderip orada derler ve çalıştırır.

- Fizik ve duyular `world.py` içinde, çizimden bağımsız tutulmalıdır.
- Eğitim pygame, pencere ya da ekran sunucusu gerektirmemelidir.
- Görüntüleyici ağ ağırlıklarını değiştirmez; fizik eğitimle aynıdır.
- Gözlemlere gizli yiyecek koordinatları eklemeyin.
- Kontroller: `python -m pytest`, `ruff check .`, `ruff format --check .`.
