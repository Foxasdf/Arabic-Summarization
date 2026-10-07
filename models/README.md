# مجلد نماذج الذكاء الاصطناعي (AI Models)

يحتوي هذا المجلد على ملفات النماذج المحلية بصيغة GGUF لتشغيل النظام بشكل محلي كامل بدون اتصال بالإنترنت.

---

## النماذج الأساسية المعتمدة للمشروع:

### 1. نموذج التلخيص الأساسي (Primary LLM):
- **النموذج**: `Falcon-H1-7B-Instruct (IQ4_XS)`
- **المصدر**: [unsloth/Falcon-H1-7B-Instruct-GGUF](https://huggingface.co/unsloth/Falcon-H1-7B-Instruct-GGUF)
- **اسم الملف**: `Falcon-H1-7B-Instruct-IQ4_XS.gguf`
- **الحجم**: ~3.88 GB
- **المعمارية**: Mamba-Transformer Hybrid — دقة فائقة وسياق طويل (256K).

### 2. نموذج تحويل الصوت إلى نص الأساسي (Primary ASR):
- **النموذج**: `Audar-ASR-V1-Turbo (Q8_0)`
- **المصدر**: [audarai/Audar-ASR-V1-Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- **ملف النموذج**: `Audar-ASR-V1-Turbo-Q8_0.gguf` (~2.01 GB)
- **ملف مشفر الصوت (Projector)**: `mmproj-Audar-ASR-V1-Turbo.gguf` (~612 MB) - *مطلوب لتشفير موجات الصوت*
- **المميزات**: المصنف الأول في دقة اللهجات العربية الفصحى والمحلية (الخليجية، المصرية، الشامية، وغيرها).

---

## التحميل التلقائي:
قم بتشغيل السكربت المخصص مع ميزة الاستئناف التلقائي في حال انقطاع الاتصال:
```bash
python scripts/download_models.py
```

أو التحقق من اكتمال النماذج:
```bash
python scripts/download_models.py --check
```
