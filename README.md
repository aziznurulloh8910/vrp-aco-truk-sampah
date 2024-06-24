Optimalisasi dan pencarian Rute terbaik

input :
- koordinat depot dan bak sampah, dalam format longitude dan latitude.
- jumlah truk
- kapasitas bak sampah (m^3)

proses :
untuk mencari rute optimal pengangkutan sampah ini menggunakan Ant Colony Optimization (ACO) pada Vehicle Routing Problem(VRP), dengan mempertimbangkan faktor berikut
- jarak terdekat (km)
- kapasitas bak sampah
- jumlah truk yang digunakan diasumsikan semua truk sama (homogen), sehingga kapasitas angkut setiap truk sama.
- jumlah emisi karbon (kg/km) 
- jumlah bensin yang dihabiskan (liter)
- biaya bahan bakar (rp)


output : 
program tersebut adalah animasi atau visualisasi bergerak dari rute terbaik dari perjalanan truk pengangkut sampah. mulai dari depot atau bisa diasumsikan sebagai TPS.
Setiap Truk keliling mengangkut sampah dari bak-bak sampah yang ada di pinggir jalan, bak-bak sampah ini diasumsikan sebagai node. 
rute yang dihasilkan digambarkan dengan garis. Dimana setiap truk digambarkan dengan warna yang berbeda - beda.

selain visualisasi, ouputnya juga menampilkan
- jarak optimal
- jumlah kubik sampah
- jumlah emisi karbon
- jumlah truk optimal
- jumlah bensin
- biaya bahan bakar (rp)