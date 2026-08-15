import re
from typing import Optional, Tuple


class GuardrailService:
    EMERGENCY_KEYWORDS = [
        "qon ketish", "qon ketayapti", "hushdan ketish", "hushimni yo'qotdim",
        "chidalmas og'riq", "shishib ketdi", "yuzim shishdi", "isitma 39",
        "isitma 40", "tez yordam", "103"
    ]

    PRESCRIPTION_KEYWORDS = [
        "dori", "antibiotik", "tavsiya qiling", "yozib bering", "retsept",
        "tashxis", "nima ichay", "nima dori", "analgin", "ketanov", "amoksitsillin"
    ]

    HUMAN_OPERATOR_KEYWORDS = [
        "odam bilan gaplashaman", "operator bilan gaplashaman", "operator",
        "tirishmang odam chaqiring", "javob bera olmadingiz", "insonga bering",
        "odam bormi"
    ]

    # Plain-text versions of responses (no HTML tags in business logic).
    # Callers are responsible for formatting / escaping before sending.
    EMERGENCY_RESPONSE_UZ = (
        "⚠️ DIQQAT! SHOSHILINCH HOLAT:\n"
        "Siz ko'rsatgan alomatlar (shoshilinch yallig'lanish, kuchli qon ketishi yoki shish) "
        "zudlik bilan shifokor yordamini talab qiladi!\n"
        "Iltimos, zudlik bilan 103 Tez Yordam xizmatiga qo'ng'iroq qiling yoki "
        "eng yaqin shoshilinch klinikaga murojaat qiling!"
    )

    PRESCRIPTION_RESPONSE_UZ = (
        "Tushunaman, lekin men klinika administratoriman va tibbiy retsept hamda "
        "dori vositalarini tavsiya qila olmayman.\n"
        "Dori vositalarini nojo'ya ta'sirlarsiz to'g'ri qabul qilish uchun shifokor ko'rigi shart.\n"
        "Sizni malakali stomatolog shifokorimiz qabuliga yozib qo'yaymi?"
    )

    OPERATOR_ESCALATION_RESPONSE_UZ = (
        "Tushundim! Men klinikamizning katta administratoriga xabaringizni uzatmoqdaman. "
        "Biroz kutishingizni so'raymiz, siz bilan bog'lanishadi."
    )

    @classmethod
    def check_guardrails(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Scans text for safety filters.
        Returns Tuple[Action, PlainTextResponse].
        Action can be: 'EMERGENCY', 'PRESCRIPTION_BLOCK', 'OPERATOR_ESCALATION', or None.
        """
        clean_text = text.lower().strip()

        # 1. Check for Operator Escalation request
        for kw in cls.HUMAN_OPERATOR_KEYWORDS:
            if kw in clean_text:
                return ("OPERATOR_ESCALATION", cls.OPERATOR_ESCALATION_RESPONSE_UZ)

        # 2. Check for Emergency Symptoms
        for kw in cls.EMERGENCY_KEYWORDS:
            if kw in clean_text:
                return ("EMERGENCY", cls.EMERGENCY_RESPONSE_UZ)

        # 3. Check for Prescription / Diagnosis request
        for kw in cls.PRESCRIPTION_KEYWORDS:
            if kw in clean_text:
                return ("PRESCRIPTION_BLOCK", cls.PRESCRIPTION_RESPONSE_UZ)

        return (None, None)
