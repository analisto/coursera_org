# Azeri Handwriting Detection Dataset

## Overview

This dataset contains 12 handwritten Azerbaijani document samples with transcriptions, representing diverse real-world document types. The dataset serves as a pilot collection for developing and validating the Azeri Handwriting Recognition (HTR) system.

**Dataset Statistics:**
- **Total Documents:** 12
- **Total Lines:** 80 (avg: 6.7 lines/document)
- **Total Characters:** 2,384 (avg: 198.7 chars/document)
- **Language:** Azerbaijani (Latin script)
- **Format:** HEIC images + TXT transcriptions
- **Created:** December 13, 2025

---

## Directory Structure

```
data/
├── images/          # 12 HEIC image files
│   ├── 01.HEIC  →  az_formal_letter_01.txt
│   ├── 02.HEIC  →  az_handwritten_note_02.txt
│   ├── 03.HEIC  →  az_numeric_mixed_03.txt
│   ├── 04.HEIC  →  az_medical_form_04.txt
│   ├── 05.HEIC  →  az_utility_application_05.txt
│   ├── 06.HEIC  →  az_bank_statement_06.txt
│   ├── 07.HEIC  →  az_education_text_07.txt
│   ├── 08.HEIC  →  az_address_list_08.txt
│   ├── 09.HEIC  →  az_technical_report_09.txt
│   ├── 10.HEIC  →  az_contract_clause_10.txt
│   ├── 11.HEIC  →  az_daily_diary_11.txt
│   └── 12.HEIC  →  az_tabular_text_12.txt
├── labels/          # 12 transcription files
│   └── az_*_##.txt
└── README.md        # This file
```

**Naming Convention:** The number at the end of each label filename (e.g., `_01`, `_12`) corresponds exactly to the image number (e.g., `01.HEIC`, `12.HEIC`).

---

## Document Types

The dataset contains 12 diverse document types representing real-world Azerbaijani documents:

| ID | Document Type | Lines | Chars | Description |
|----|---------------|-------|-------|-------------|
| 01 | **Formal Letter** | 9 | 196 | Official business letter to director about 2025 reports |
| 02 | **Handwritten Note** | 8 | 171 | Personal reminder about bank appointment |
| 03 | **Numeric Mixed** | 6 | 147 | Contract with numbers, dates, amounts (VAT calculation) |
| 04 | **Medical Form** | 8 | 215 | Patient form with diagnosis and prescriptions |
| 05 | **Utility Application** | 8 | 251 | Electricity complaint with meter reading |
| 06 | **Bank Statement** | 6 | 165 | Account transactions with debits/credits |
| 07 | **Education Text** | 6 | 240 | Constitutional text about education rights |
| 08 | **Address List** | 7 | 184 | Addresses from Baku, Ganja, Sumqayit cities |
| 09 | **Technical Report** | 4 | 216 | System performance analysis (CPU, disk) |
| 10 | **Contract Clause** | 8 | 257 | Legal contract clause text |
| 11 | **Daily Diary** | 6 | 184 | Personal diary entry about work project |
| 12 | **Tabular Text** | 4 | 158 | Employee table with names, ages, departments |

### Complexity Levels

**Simple (4-6 lines):**
- Tabular text (12), Technical report (09), Numeric mixed (03)
- Short, structured content

**Medium (6-8 lines):**
- Bank statement (06), Education (07), Address list (08)
- Moderate length with mixed content

**Complex (8-9 lines):**
- Medical form (04), Handwritten note (02), Contract (10)
- Longer documents with varied formatting

---

## Image Characteristics

**Format:** HEIC (High Efficiency Image Container)
- **Codec:** HEVC (H.265) - Apple's modern image format
- **File Sizes:** 820KB - 1.0MB per image (average: ~890KB)
- **Total Size:** ~10.5MB

**Important:** HEIC format requires conversion to PNG/JPG for PyTorch processing:

```python
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()
img = Image.open('01.HEIC').convert('L')  # Convert to grayscale
img.save('01.png')
```

---

## Label File Format

**Format:**
```
Line_number→Transcribed_text
```

**Example (az_tabular_text_12.txt):**
```
     1→Adı        | Yaşı | Şöbə
     2→-------------------------
     3→Rauf       | 32   | Maliyyə
     4→Aysel      | 28   | İnsan resursları
     5→Kamal      | 41   | İT dəstəyi
```

**Characteristics:**
- Line numbers with leading spaces
- Arrow delimiter (`→`) separates line number from text
- Preserves original spacing and formatting
- Includes punctuation and special characters exactly as written

---

## Azerbaijani Language Statistics

### Character Distribution (Top 15)

```
Space:  262 occurrences (11% - word boundaries)
a:      154 (6.5%)
i:      137 (5.7%)
ə:      125 (5.2%) ← Azerbaijani-specific schwa
r:       98 (4.1%)
l:       91 (3.8%)
n:       90 (3.8%)
ı:       66 (2.8%) ← Azerbaijani dotless i
s:       66 (2.8%)
m:       57 (2.4%)
d:       51 (2.1%)
t:       49 (2.1%)
e:       42 (1.8%)
u:       39 (1.6%)
-:       35 (1.5%) ← Hyphenation
```

### Azerbaijani-Specific Characters (Critical)

**Lowercase:**
```
ə: 125  (schwa - most common special character)
ı:  66  (dotless i)
ş:  27  (s with cedilla)
ü:  27  (u with diaeresis)
ğ:  13  (g with breve)
ö:   7  (o with diaeresis)
ç:   7  (c with cedilla)
```

**Uppercase:**
```
Ə:   4
İ:   3  (i with dot - Turkish/Azeri uppercase)
Ş:   2
Ü:   1
Ö:   1
```

**Total Azerbaijani-specific characters:** 263 (11% of all characters)

**Key Insight:** Azerbaijani diacritics (ə, ı, ş, ü, ğ, ö, ç) are **essential** and must be preserved by the HTR model.

---

## Content Analysis

### Character Breakdown

```
Letters (a-z, A-Z):  ~1,550 (65%)
Azerbaijani chars:      263 (11%)
Spaces:                 262 (11%)
Numbers:                ~150 (6%)
Punctuation:            ~160 (7%)
```

### Numeric & Special Content

**Numbers Present:**
- Dates: `14.06.2024`, `01.02.2025`, `03.11.1987`
- Amounts: `12 750.45 AZN`, `15 045.53 AZN`, `304.50 AZN`
- Percentages: `18%`, `67%`
- Phone numbers: `050-3456789`
- Contract numbers: `№ 457/23`
- Account numbers: `AZ21NABZ`

**Punctuation & Symbols:**
- Hyphens (35) - word breaks, line wrapping
- Periods (34) - decimals, abbreviations
- Commas (19) - number separators
- Colons (15) - field labels
- Pipe symbols (|) - table formatting

---

## Domain-Specific Vocabulary

The dataset contains rich domain terminology across multiple sectors:

**Financial:**
- `müqavilə` (contract), `məbləğ` (amount), `ƏDV` (VAT)
- `hesabat` (report), `hesab nömrəsi` (account number)
- `Maaş` (salary), `Komunal` (utilities)

**Medical:**
- `pasiyent` (patient), `diaqnoz` (diagnosis)
- `baş ağrısı` (headache), `arterial hipertenziya` (hypertension)
- `dərman` (medicine)

**Legal/Formal:**
- `Hörmətli` (Dear/Honorable), `direktor` (director)
- `Konstitusiya` (Constitution), `qanunvericilik` (legislation)
- `qarşılıqlı razılaşma` (mutual agreement)

**Technical:**
- `sistem performans` (system performance)
- `CPU yüklənməsi` (CPU load), `disk oxunuş` (disk read)

**Personal/General:**
- `xahiş edirəm` (I request), `bildiririk` (we inform)
- `ünvan` (address), `rayon` (district)

---

## Text Features & Challenges

### Line Breaking & Hyphenation

Multiple documents show **mid-word line breaks** with hyphens:
- `hazır-lanması` (prepared, split across lines)
- `yaran-mış` (arising)
- `araşdırıl-masını` (investigation)
- `hiper-tenziya` (hypertension)

**Implication:** HTR model needs line-level recognition, and post-processing must reconstruct hyphenated words.

### Tabular Formatting

Document 12 contains **table structure**:
```
Adı        | Yaşı | Şöbə
-------------------------
Rauf       | 32   | Maliyyə
Aysel      | 28   | İnsan resursları
```

**Implication:** Stage 1 (layout detection) is critical for preserving table structure.

### Mixed Case Usage

- **Proper nouns:** `Bakı`, `Gəncə`, `Neftçilər prospekti`
- **Abbreviations:** `ƏDV`, `AZN`, `ATM`, `CPU`, `IT`
- **Sentence case:** Standard for regular text

---

## Data Quality Assessment

### Strengths

✅ **Diverse document types** - Covers real-world use cases across multiple domains

✅ **Rich Azerbaijani vocabulary** - Proper diacritics preserved throughout

✅ **Mixed content** - Text, numbers, tables, addresses

✅ **Domain variety** - Medical, legal, financial, technical, personal

✅ **Proper formatting** - Line-level transcriptions with structure preservation

✅ **Clean transcriptions** - Accurate character-level annotations

### Limitations

⚠️ **Small dataset size** - Only 12 samples (insufficient for production training)

⚠️ **No writer diversity info** - Unknown if single/multiple writers

⚠️ **HEIC format** - Requires preprocessing for PyTorch

⚠️ **No bounding boxes** - Labels are page-level, not line-level

⚠️ **No validation split** - Need to define train/val/test splits

⚠️ **No image metadata** - Resolution, DPI, quality information missing

⚠️ **Insufficient for LM training** - Only 2.4K chars vs. 100K-1M recommended

---

## Preprocessing Requirements

Before training, the following preprocessing steps are required:

### 1. Convert HEIC to PNG/JPG

```python
import pillow_heif
from PIL import Image
import os

pillow_heif.register_heif_opener()

for heic_file in os.listdir('images/'):
    if heic_file.endswith('.HEIC'):
        img_path = os.path.join('images/', heic_file)
        img = Image.open(img_path).convert('L')  # Grayscale
        png_path = img_path.replace('.HEIC', '.png')
        img.save(png_path)
```

### 2. Line Segmentation

Extract bounding boxes for each line from page images:
- Use layout detection (YOLOv8) or manual annotation
- Create line-level image crops
- Map each line crop to its transcription

### 3. Create Vocabulary File

```python
import json

# Extract all unique characters from labels
chars = set()
for label_file in label_files:
    with open(label_file, 'r', encoding='utf-8') as f:
        text = f.read()
        # Remove line numbers and arrow delimiter
        text = '→'.join(text.split('→')[1:]) if '→' in text else text
        chars.update(text)

# Create vocabulary mapping
vocab = {char: idx for idx, char in enumerate(sorted(chars))}
vocab['[BLANK]'] = len(vocab)  # CTC blank token

with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
```

### 4. Define Data Splits

Recommended split (document-wise to prevent data leakage):

```
train.txt: 01,02,03,04,05,06,07,08,09  (75% - 9 documents)
val.txt:   10,11                        (17% - 2 documents)
test.txt:  12                           (8% - 1 document)
```

---

## Recommended Character Set

Based on dataset analysis, the vocabulary should include:

**Latin Letters:**
- Lowercase: a-z
- Uppercase: A-Z

**Azerbaijani Characters:**
- Lowercase: ə, ç, ğ, ı, ö, ş, ü
- Uppercase: Ə, Ç, Ğ, İ, Ö, Ş, Ü

**Digits:** 0-9

**Punctuation & Symbols:**
```
. , : ; - – — ( ) [ ] / |
" ' « » ? ! № % + =
```

**Special:**
- Space character
- CTC blank token

**Estimated Vocabulary Size:** ~100 characters

---

## Usage Guidelines

### For Model Training

1. **Data Augmentation is Critical** - With only 12 samples, heavy augmentation is mandatory:
   - Rotation: ±3°
   - Scaling: 0.9-1.1
   - Elastic distortion
   - Blur, noise, random erasing
   - Synthetic overlays

2. **Start with HTR-Lite** - Use the lightweight model variant for proof-of-concept

3. **Character-Level Tokenization** - Recommended for Azerbaijani language

4. **Preserve Diacritics** - Critical for maintaining language integrity

### For Dataset Expansion

**Immediate Actions:**

1. **Collect More Data** - Current 2.4K chars is far below recommended 100K-1M
   - Photograph additional handwritten documents
   - Use synthetic data generation
   - Apply pseudo-labeling on unlabeled scans

2. **Line-Level Annotation** - Convert page-level to line-level:
   - Extract individual line bounding boxes
   - Crop and save as separate line images
   - Create line-level transcriptions

3. **Metadata Collection** - Document:
   - Writer information (for stratified splits)
   - Image resolution and DPI
   - Document quality scores
   - Date of collection

4. **Quality Control** - Verify:
   - Transcription accuracy
   - Diacritic correctness
   - Proper character encoding (UTF-8)

---

## Integration with Architecture

### Alignment with Planned System (plan.md)

| Planned Feature | Current Data Status |
|----------------|---------------------|
| **Character-level vocab** | ✅ Azerbaijani chars present |
| **Document-wise split** | ⚠️ Not defined yet |
| **Line-level images** | ❌ Only page-level currently |
| **Bounding boxes** | ❌ Not annotated |
| **Mixed content** | ✅ Numbers, text, tables |
| **Domain diversity** | ✅ 12 document types |
| **100K+ tokens for LM** | ❌ Only 2.4K chars |

### Next Steps for Implementation

1. **Preprocessing Pipeline:**
   - Convert HEIC → PNG
   - Segment pages into lines
   - Extract line bounding boxes

2. **Dataset Preparation:**
   - Create train/val/test splits
   - Generate vocabulary.json
   - Build data loader with augmentation

3. **Baseline Model:**
   - Train HTR-Lite on augmented data
   - Evaluate on validation set
   - Analyze error patterns

4. **Data Expansion:**
   - Collect 100+ more documents
   - Implement active learning loop
   - Build Azerbaijani language model corpus

---

## Example Label Samples

### Document 01 - Formal Letter
```
Hörmətli cənab direktor,
Bu məktub vasitəsilə
bildiririk ki,
2025-ci il
üzrə hesabatların
hazırlanması başa çatmaq
üzrədir.
```

### Document 04 - Medical Form
```
Pasiyentin adı, soyadı:
Əliyev Rəşad Kamran oğlu
Doğum tarixi: 03.11.1987
Şikayətlər: baş ağrısı, halsızlıq,
yuxusuzluq
Diaqnoz: arterial hipertenziya
```

### Document 12 - Tabular Text
```
Adı        | Yaşı | Şöbə
-------------------------
Rauf       | 32   | Maliyyə
Aysel      | 28   | İnsan resursları
Kamal      | 41   | İT dəstəyi
```

---

## References

- **Project Plan:** See `../plan.md` for full architecture specification
- **Character Encoding:** UTF-8
- **Language:** Azerbaijani (Latin script, ISO 639-1: az)
- **Image Format:** HEIC (requires conversion to PNG/JPG)

---

## License & Usage

This dataset is collected for developing the Azeri Handwriting Detection system. Please ensure proper handling of any personal information that may appear in the documents.

---

**Last Updated:** December 13, 2025
**Dataset Version:** 1.0 (Pilot)
**Total Samples:** 12 documents, 80 lines, 2,384 characters
