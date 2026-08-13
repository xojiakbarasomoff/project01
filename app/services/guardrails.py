import re
from typing import Dict, Any, Optional, Tuple


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

    EMERGENCY_RESPONSE_UZ = (
        "⚠️ <b>DIQQAT! SHOSHILINCH HOLAT:</b><br/>"
        "Siz ko'rsatgan alomatlar (shoshilinch yallig'lanish, kuchli qon ketishi yoki shish) "
        "zudlik bilan tibbiy yordam talab qiladi!<br/>"
        "Iltimos, zudlik bilan <b>103 Tez Yordam</b> xizmatiga qo'ng'iroq qiling yoki eng yaqin shoshilinch klinikaga murojaat qiling!"
    )

    PRESCRIPTION_RESPONSE_UZ = (
        "Tushunaman, lekin men AI-yordamchiman va tibbiy retsept hamda dori vositalarini tavsiya qila olmayman.<br/>"
        "Dori vositalarini nojo'ya ta'sirlarsiz to'g'ri qabul qilish uchun shifokor ko'rigi shart.<br/>"
        "Sizni malakali stomatologimiz qabuliga yozib qo'yaymi?"
    )

    OPERATOR_ESCALATION_RESPONSE_UZ = (
        "Tushundim! Chatni klinika operatoriga uzatmoqdaman. Biroz kutishingizni so'raymiz, "
        "operatorimiz tez orada siz bilan bog'lanadi."
    )

    @classmethod
    def check_guardrails(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Scans text for safety filters.
        Returns Tuple[Action, CustomResponse]
        Action can be: 'EMERGENCY', 'PRESCRIPTION_BLOCK', 'OPERATOR_ESCALATION', or None
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
