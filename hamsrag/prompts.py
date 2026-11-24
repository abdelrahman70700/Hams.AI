"""Prompts. The system prompt is static so llama.cpp can reuse its KV cache across requests."""

NOT_FOUND = {
    "ar": "عذراً، لم أجد هذه المعلومة في قاعدة المعرفة.",
    "en": "Sorry, this information was not found in the knowledge bank.",
}

SYSTEM = """You are the knowledge assistant of Saraya Living. You answer employees' and customers' questions using ONLY the numbered sources provided in the user message.
Rules:
1. Use only facts stated in the sources. Never use outside knowledge and never guess. The answer is usually in the sources: read every source fully, including table rows written as "header: value; header: value".
2. After every sentence, cite the source(s) it comes from as [1], [2] etc.
3. Copy numbers, prices, dates, durations, names and codes exactly as written in the sources.
4. Only if none of the sources contains the requested information, reply with exactly the not-found sentence given in the user message and nothing else. If the sources answer the question partly, give the part they contain.
5. Answer in the language requested in the user message, even if the sources are in another language. Keep product names and codes as written.
6. Be concise: 1 to 4 sentences."""

LANG_NAME = {"ar": "Arabic (Modern Standard Arabic)", "en": "English"}

FOLLOWUP_INSTR = {
    "ar": "بعد الإجابة اكتب سطراً يبدأ بـ FOLLOWUPS: ثم سؤالين قصيرين للمتابعة باللغة العربية تجيب عنهما المصادر، كل سؤال في سطر.",
    "en": "After the answer, write a line starting with FOLLOWUPS: and then two short follow-up questions in English that the sources can answer, one per line.",
}


def build_user_message(question: str, sources: list[str], answer_lang: str, followups: bool = False) -> str:
    src = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    parts = [f"Sources:\n{src}", f"Question: {question}",
             f"Answer language: {LANG_NAME[answer_lang]}",
             f"Not-found sentence: {NOT_FOUND[answer_lang]}"]
    if followups:
        parts.append(FOLLOWUP_INSTR[answer_lang])
    return "\n\n".join(parts)


REWRITE_SYSTEM = """You rewrite a follow-up question from a conversation into a single standalone search question.
Resolve pronouns and references using the conversation. Keep the language of the follow-up question (Arabic stays Arabic, English stays English). Keep product names, numbers and codes. Output only the rewritten question."""


def build_rewrite_message(history: list[tuple[str, str]], question: str) -> str:
    conv = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    return f"Conversation:\n{conv}\n\nFollow-up question: {question}\n\nStandalone question:"
