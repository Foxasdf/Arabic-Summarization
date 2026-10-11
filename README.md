# ملخص الاجتماعات العربية 🎙️

تطبيق ويب محلي بالكامل لتلخيص الاجتماعات باللغة العربية. يقوم بتحويل الصوت إلى نص مكتوب مع فصل المتحدثين، ثم تلخيص الاجتماع بالنقاط المهمة.

**جميع نماذج الذكاء الاصطناعي تعمل محلياً على جهازك — لا حاجة للإنترنت أثناء الاستخدام.**

---

## ✨ المميزات

- 🎤 **تحويل الصوت إلى نص** — باستخدام نموذج Audar ASR المتخصص بالعربية
- 👥 **فصل المتحدثين** — تحديد كل متحدث تلقائياً بألوان مختلفة
- 📋 **تلخيص ذكي** — استخراج النقاط المهمة والقرارات والمهام
- 🔒 **محلي بالكامل** — بياناتك لا تغادر جهازك
- 🌐 **بث عبر الشبكة** — يمكن لأي جهاز على الشبكة المحلية الوصول
- 💻 **متعدد المنصات** — يعمل على Windows و Linux
- 🔄 **قابل للتوسيع** — إضافة نماذج LLM جديدة بسهولة

## 📦 النماذج المستخدمة

| النموذج | الوظيفة | الحجم |
|---------|---------|-------|
| [Falcon-H1-Arabic 7B](https://huggingface.co/tiiuae/Falcon-H1-Arabic-7B-Instruct-GGUF) | تلخيص الاجتماعات | ~5 GB |
| [Audar-ASR-V1-Flash](https://huggingface.co/audarai/Audar-ASR-V1-Flash-GGUF) | تحويل الصوت لنص | ~500 MB |
| [pyannote.audio 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | فصل المتحدثين | ~100 MB |

## 🖥️ متطلبات النظام

- **Python** 3.10+
- **Node.js** 18+
- **ffmpeg** (لمعالجة الصوت)
- **GPU** (موصى): NVIDIA GPU بـ 8+ GB VRAM
- **RAM**: 16 GB كحد أدنى (للتشغيل على CPU)

## 🚀 التشغيل السريع

### 1. تحميل النماذج

```bash
pip install huggingface-hub
python scripts/download_models.py
```

### 2. التشغيل

**Windows:**
```bash
scripts\start.bat
```

**Linux:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Docker:**
```bash
docker-compose up --build
```

### 3. الاستخدام

افتح المتصفح على `http://localhost:3011`

للوصول من أجهزة أخرى على الشبكة: `http://[عنوان-IP-جهازك]:3011`

## 📁 هيكل المشروع

```
├── backend/          # خادم FastAPI (Python)
│   ├── app/
│   │   ├── api/      # نقاط API
│   │   ├── models/   # محركات الذكاء الاصطناعي
│   │   ├── services/ # خدمات المعالجة
│   │   └── schemas/  # نماذج البيانات
│   └── requirements.txt
├── frontend/         # واجهة المستخدم (React + TypeScript)
│   └── src/
├── models/           # ملفات النماذج (GGUF)
├── scripts/          # سكربتات التشغيل والتحميل
└── docker-compose.yml
```

## 🔧 إضافة نموذج LLM جديد

1. حمّل ملف GGUF إلى مجلد `models/`
2. أضف التسجيل في `backend/app/models/registry.py`:

```python
registry.register("model-key", ModelInfo(
    name="اسم النموذج",
    model_type="llm",
    gguf_filename="filename.gguf",
    description="وصف النموذج",
))
```

3. أعد تشغيل الخادم — يظهر تلقائياً في الواجهة

## 🔄 سير العمل

```
رفع ملف صوتي → معالجة الصوت (ffmpeg) → تحويل لنص (ASR)
                                       → فصل المتحدثين (pyannote)
                                       → مطابقة النتائج → عرض النص
                                       → تلخيص (اختياري) → عرض الملخص
```

## 📝 الترخيص

هذا المشروع للاستخدام الداخلي.
