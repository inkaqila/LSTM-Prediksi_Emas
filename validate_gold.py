import pandas as pd
import matplotlib.pyplot as plt

# Baca file CSV dengan delimiter ';'
df = pd.read_csv('XAU_1d_data.csv', sep=';')

print("🔍 Info Dataset:")
print(df.info())
print("\n📋 5 Baris Pertama:")
print(df.head())

# Konversi kolom 'Date' ke datetime dan jadikan index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Validasi kolom 'Close'
if 'Close' in df.columns:
    print("\n✅ Kolom 'Close' ditemukan. Menampilkan grafik...")

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], color='gold', linewidth=1)
    plt.title('Harga Emas Harian (XAU/USD)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (USD per Troy Ounce)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"\n📅 Rentang Waktu: {df.index.min()} sampai {df.index.max()}")
    print(f"📊 Jumlah Data: {len(df)} hari")
else:
    print("\n⚠️ Kolom 'Close' tidak ditemukan! Kolom yang ada:", df.columns.tolist())