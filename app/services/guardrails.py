import re
from typing import Optional, Tuple


class GuardrailService:
    EMERGENCY_KEYWORDS_UZ = [
        "qon ketish", "qon ketayapti", "hushdan ketish", "hushimni yo'qotdim",
        "chidalmas og'riq", "shishib ketdi", "yuzim shishdi", "isitma 39",
        "isitma 40", "tez yordam", "103",
        "қон кетиш", "қон кетаяпти", "ҳушдан кетиш", "ҳушимни йўқотдим",
        "чидалмас оғриқ", "шишиб кетди", "юзим шишди", "иситма 39", "иситма 40"
    ]

    EMERGENCY_KEYWORDS_RU = [
        "кровотечение", "идти кровь", "потеря сознания", "обморок",
        "сильная боль", "опухло лицо", "отек", "температура 39",
        "температура 40", "скорая помощь"
    ]

    PRESCRIPTION_KEYWORDS_UZ = [
        "dori", "antibiotik", "tavsiya qiling", "yozib bering", "retsept",
        "tashxis", "nima ichay", "nima dori", "analgin", "ketanov", "amoksitsillin",
        "дори", "антибиотик", "тавсия қилинг", "ёзиб беринг", "рецепт",
        "ташхис", "нима ичай", "нима дори"
    ]

    PRESCRIPTION_KEYWORDS_RU = [
        "лекарство", "антибиотик", "посоветуйте", "выпишите", "рецепт",
        "диагноз", "что пить", "какие таблетки", "препарат", "назначьте"
    ]

    HUMAN_OPERATOR_KEYWORDS_UZ = [
        "odam bilan gaplashaman", "operator bilan gaplashaman", "operator",
        "tirishmang odam chaqiring", "javob bera olmadingiz", "insonga bering",
        "odam bormi", "оператор", "одам билан гаплашаман"
    ]

    HUMAN_OPERATOR_KEYWORDS_RU = [
        "оператор", "позовите человека", "живой человек", "человек",
        "соедините с оператором", "поговорить с человеком"
    ]

    RU_INDICATORS = [
        "здравствуйте", "привет", "добрый день", "цена", "стоимость",
        "где вы", "адрес", "запись", "врач", "сколько стоит", "когда",
        "пожалуйста", "как доехать", "прием"
    ]

    @staticmethod
    def detect_script(text: str) -> str:
        """Returns 'cyrillic' if text contains Cyrillic characters, else 'latin'."""
        if re.search(r'[\u0400-\u04FF]', text):
            return "cyrillic"
        return "latin"

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Returns 'ru' if text matches Russian indicators, else 'uz'."""
        clean = text.lower()
        for kw in cls.RU_INDICATORS:
            if kw in clean:
                return "ru"
        return "uz"

    @classmethod
    def get_greeting(cls, text: str) -> str:
        script = cls.detect_script(text)
        lang = cls.detect_language(text)
        if lang == "ru":
            return "Здравствуйте!"
        if script == "cyrillic":
            return "Ассалому алайкум!"
        return "Assalomu alaykum!"

    @classmethod
    def check_guardrails(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Scans text for safety filters and language/script mirroring.
        Returns Tuple[Action, PlainTextResponse].
        Action can be: 'EMERGENCY', 'PRESCRIPTION_BLOCK', 'OPERATOR_ESCALATION', or None.
        """
        clean_text = text.lower().strip()
        script = cls.detect_script(text)
        lang = cls.detect_language(text)

        # 1. Operator Escalation
        for kw in cls.HUMAN_OPERATOR_KEYWORDS_UZ + cls.HUMAN_OPERATOR_KEYWORDS_RU:
            if kw in clean_text:
                if lang == "ru":
                    msg = "Понял вас! Я передаю ваше сообщение старшему администратору клиники. Пожалуйста, подождите немного, с вами свяжутся."
                elif script == "cyrillic":
                    msg = "Тушундим! Мен клиникамизнинг катта администраторига хабарингизни узатмоқдаман. Бироз кутишингизни сўраймиз, сиз билан боғланишади."
                else:
                    msg = "Tushundim! Men klinikamizning katta administratoriga xabaringizni uzatmoqdaman. Biroz kutishingizni so'raymiz, siz bilan bog'lanishadi."
                return ("OPERATOR_ESCALATION", msg)

        # 2. Emergency Symptoms
        for kw in cls.EMERGENCY_KEYWORDS_UZ + cls.EMERGENCY_KEYWORDS_RU:
            if kw in clean_text:
                if lang == "ru":
                    msg = ("⚠️ ВНИМАНИЕ! ЭКСТРЕННАЯ СИТУАЦИЯ:\n"
                           "Указанные вами симптомы требуют немедленного осмотра врача!\n"
                           "Пожалуйста, срочно позвоните в службу Скорой Помощи 103 или обратитесь в ближайшую клинику!")
                elif script == "cyrillic":
                    msg = ("⚠️ ДИҚҚАТ! ШОШИЛИНЧ ҲОЛАТ:\n"
                           "Сиз кўрсатган аломатлар зудлик билан шифокор ёрдамини талаб қилади!\n"
                           "Илтимос, зудлик билан 103 Тез Ёрдам хизматига қўнғироқ қилинг ёки энг яқин шошилинч клиникага мурожаат қилинг!")
                else:
                    msg = ("⚠️ DIQQAT! SHOSHILINCH HOLAT:\n"
                           "Siz ko'rsatgan alomatlar zudlik bilan shifokor yordamini talab qiladi!\n"
                           "Iltimos, zudlik bilan 103 Tez Yordam xizmatiga qo'ng'iroq qiling yoki eng yaqin shoshilinch klinikaga murojaat qiling!")
                return ("EMERGENCY", msg)

        # 3. Prescription / Treatment advice request
        for kw in cls.PRESCRIPTION_KEYWORDS_UZ + cls.PRESCRIPTION_KEYWORDS_RU:
            if kw in clean_text:
                if lang == "ru":
                    msg = ("Я понимаю, но я являюсь администратором клиники и не могу назначать лекарства или ставить диагноз.\n"
                           "Для назначения лечения необходим осмотр врача. Записать вас на прием к нашему врачу?")
                elif script == "cyrillic":
                    msg = ("Тушунаман, лекин мен клиника администраториман ва тиббий рецепт ҳамда дори воситаларини тавсия қила олмайман.\n"
                           "Дори воситаларини тўғри қабул қилиш учун шифокор кўриги шарт. Сизни шифокоримиз қабулига ёзиб қўяйми?")
                else:
                    msg = ("Tushunaman, lekin men klinika administratoriman va tibbiy retsept hamda dori vositalarini tavsiya qila olmayman.\n"
                           "Dori vositalarini to'g'ri qabul qilish uchun shifokor ko'rigi shart. Sizni shifokorimiz qabuliga yozib qo'yaymi?")
                return ("PRESCRIPTION_BLOCK", msg)

        return (None, None)

