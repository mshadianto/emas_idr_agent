# agent_setup.py
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Impor tools dari file tools.py
from tools import all_tools

def create_agent_executor():
    """
    Membuat agent executor yang HANYA menggunakan model lokal via Ollama.
    """
    # 1. Inisialisasi LLM lokal (pastikan Ollama & model Qwen2 sudah siap)
    try:
        llm = ChatOllama(model="qwen2:7b", temperature=0)
    except Exception as e:
        # Menambahkan error handling jika Ollama tidak berjalan
        raise ConnectionError(f"Tidak dapat terhubung ke Ollama. Pastikan Ollama sudah berjalan di background. Detail error: {e}")


    # 2. Definisikan Prompt Template (TIDAK BERUBAH)
    template = """
Anda adalah EmasIDR Agent, seorang analis keuangan AI yang ahli dari Indonesia.
Anda menganalisis harga emas dan kurs USD/IDR. Jawab pertanyaan pengguna dalam Bahasa Indonesia.

Anda memiliki akses ke tools berikut:
{tools}

Gunakan format berikut untuk berpikir:
Pertanyaan: Pertanyaan awal dari pengguna
Pikiran: Anda harus berpikir tentang apa yang harus dilakukan untuk menjawab pertanyaan ini.
Aksi: Aksi yang akan diambil, harus salah satu dari [{tool_names}].
Input Aksi: Input untuk aksi tersebut.
Observasi: Hasil dari eksekusi aksi.
... (Pikiran, Aksi, Input Aksi, Observasi ini bisa berulang jika diperlukan)
Pikiran: Saya sekarang memiliki informasi yang cukup untuk menjawab pertanyaan pengguna.
Jawaban Akhir: Jawaban final untuk pertanyaan awal pengguna, disampaikan secara jelas dan komprehensif dalam Bahasa Indonesia.

Mulai!

Pertanyaan: {input}
Pikiran:{agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(template)

    # 3. Buat Agent (TIDAK BERUBAH)
    agent = create_react_agent(llm, all_tools, prompt)

    # 4. Buat Agent Executor (TIDAK BERUBAH)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor