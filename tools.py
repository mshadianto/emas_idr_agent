# tools.py
import requests
from bs4 import BeautifulSoup
from langchain.agents import tool

# Impor tool baru dari LangChain Community
from langchain_community.tools import DuckDuckGoSearchRun

# Tool 1: Mengambil harga emas (TIDAK BERUBAH)
@tool
def get_antam_gold_price() -> str:
    """Mengambil harga beli emas Antam 1 gram terbaru dari situs Logam Mulia. Gunakan ini untuk mendapatkan data harga emas terkini."""
    try:
        url = "https://www.logammulia.com/id/harga-emas-hari-ini"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        price_element = soup.find('td', string='1 gr').find_next_sibling('td')
        price = price_element.get_text(strip=True)
        return f"Harga beli emas Antam 1 gram per hari ini adalah: {price}"
    except Exception as e:
        return f"Gagal mengambil data harga emas: {e}. Mungkin ada perubahan pada situs web sumber."

# Tool 2: Mengambil kurs USD/IDR (TIDAK BERUBAH)
@tool
def get_usd_idr_rate() -> str:
    """Mengambil kurs USD ke IDR terkini. Gunakan ini untuk mendapatkan nilai tukar Dolar AS terhadap Rupiah."""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data['rates']['IDR']
        return f"Kurs USD/IDR saat ini adalah: Rp {rate:,.2f}"
    except Exception as e:
        return f"Gagal mengambil data kurs: {e}"

# Tool 3: Pencarian Web (DIGANTI DENGAN DUCKDUCKGO)
# DuckDuckGoSearchRun tidak memerlukan API key.
web_search_tool = DuckDuckGoSearchRun()
web_search_tool.description = "Alat pencarian web untuk mencari berita, peristiwa, atau informasi terkini. Gunakan untuk pertanyaan tentang sentimen pasar, berita ekonomi, atau konteks peristiwa."

# Daftar semua tools yang akan diekspor (diperbarui dengan tool baru)
all_tools = [get_antam_gold_price, get_usd_idr_rate, web_search_tool]