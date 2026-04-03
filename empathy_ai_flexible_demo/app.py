from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import librosa
import numpy as np

APP_DIR = Path(__file__).resolve().parent
CASCADE_DIR = Path(cv2.data.haarcascades)
FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_DIR / "haarcascade_frontalface_default.xml"))
SMILE_CASCADE = cv2.CascadeClassifier(str(CASCADE_DIR / "haarcascade_smile.xml"))
EYE_CASCADE = cv2.CascadeClassifier(str(CASCADE_DIR / "haarcascade_eye.xml"))

EMOTION_LABELS = [
    "vui",
    "buồn",
    "giận",
    "lo_lắng",
    "căng_thẳng",
    "thất_vọng",
    "mệt_mỏi",
    "trung_tính",
    "không_xác_định",
]

EMOTION_ALIASES: Dict[str, str] = {
    "vui": "Vui vẻ",
    "buồn": "Buồn",
    "giận": "Tức giận",
    "lo_lắng": "Lo lắng",
    "căng_thẳng": "Căng thẳng",
    "thất_vọng": "Thất vọng",
    "mệt_mỏi": "Mệt mỏi",
    "trung_tính": "Trung tính",
    "không_xác_định": "Không xác định",
}

EMOJI = {
    "vui": "🟢",
    "buồn": "🔵",
    "giận": "🔴",
    "lo_lắng": "🟠",
    "căng_thẳng": "🟠",
    "thất_vọng": "🟣",
    "mệt_mỏi": "🟤",
    "trung_tính": "⚪",
    "không_xác_định": "⚫",
}

SOURCE_LABELS = {
    "text": "Văn bản",
    "face": "Khuôn mặt",
    "audio": "Âm thanh",
}
SOURCE_DISPLAY_TO_KEY = {
    "Văn bản": "text",
    "Khuôn mặt": "face",
    "Âm thanh": "audio",
}

TEXT_EMOTION_LEXICON: Dict[str, List[str]] = {
    "vui": [
        "vui", "hạnh phúc", "tuyệt", "ổn", "đã xong", "hoàn thành", "mừng", "hào hứng", "phấn khởi",
        "rất tốt", "thật tốt", "đỡ rồi", "thành công", "yay", "happy", "great",
    ],
    "buồn": [
        "buồn", "chán", "cô đơn", "trống rỗng", "tụt mood", "nản", "suy", "khóc", "mất động lực",
        "không vui", "sad", "down",
    ],
    "giận": [
        "giận", "bực", "ức", "khó chịu", "tức", "cay", "bất mãn", "ghét", "điên tiết", "annoyed", "angry",
    ],
    "lo_lắng": [
        "lo", "sợ", "hồi hộp", "băn khoăn", "hoang mang", "áp lực", "deadline", "phỏng vấn", "thi", "nervous", "worry",
    ],
    "căng_thẳng": [
        "stress", "căng", "ngộp", "rối", "quá tải", "bí", "mắc kẹt", "không xoay được", "gồng", "cháy deadline",
    ],
    "thất_vọng": [
        "thất vọng", "không đạt", "bị chê", "bị phê bình", "bị mắng", "không như mong đợi", "tụt hẫng",
    ],
    "mệt_mỏi": [
        "mệt", "đuối", "kiệt sức", "uể oải", "buồn ngủ", "mất ngủ", "rã rời", "tired", "exhausted",
    ],
}

NEGATIONS = {"không", "chẳng", "chưa", "chả", "đừng"}
INTENSIFIERS = {"rất", "quá", "cực kỳ", "siêu", "khá", "hơi", "thật sự", "vô cùng"}

CONCERN_RULES: Dict[str, List[str]] = {
    "học tập": ["bài", "môn", "thi", "điểm", "deadline", "báo cáo", "luận văn", "đồ án", "học"],
    "công việc": ["công việc", "sếp", "đồng nghiệp", "dự án", "meeting", "task", "kpi"],
    "quan hệ": ["gia đình", "bạn bè", "người yêu", "thầy", "cô", "mâu thuẫn", "cãi nhau"],
    "tự tin bản thân": ["không đủ tốt", "tự ti", "mặc cảm", "thất bại", "kém", "không làm được"],
    "sức khỏe": ["đau", "mất ngủ", "mệt", "kiệt sức", "ốm", "không khỏe"],
}

RESPONSE_BANK: Dict[str, Dict[str, List[str]]] = {
    "vui": {
        "open": [
            "Mình cảm nhận được bạn đang có nguồn năng lượng rất tích cực.",
            "Nghe bạn chia sẻ là mình thấy niềm vui đang hiện rõ rồi.",
        ],
        "follow": [
            "Đó là thành quả đáng ghi nhận. Bạn muốn giữ đà này bằng việc chốt nốt bước tiếp theo không?",
            "Khoảnh khắc này rất đáng trân trọng. Bạn có thể ghi lại điều gì đã giúp bạn làm tốt để lặp lại sau này.",
        ],
    },
    "buồn": {
        "open": [
            "Mình hiểu là lúc này bạn đang khá nặng lòng.",
            "Nghe vậy mình cảm nhận rõ sự buồn bã trong câu chuyện của bạn.",
        ],
        "follow": [
            "Bạn không cần tự gồng hết một mình. Mình có thể cùng bạn tách rõ điều gì đang làm bạn buồn nhất.",
            "Mình đang ở đây để lắng nghe. Bạn muốn bắt đầu từ sự việc vừa xảy ra hay từ cảm giác của bạn lúc này?",
        ],
    },
    "giận": {
        "open": [
            "Mình thấy bạn đang rất khó chịu và bị dồn cảm xúc lên khá mạnh.",
            "Nghe như bạn đang bực thật sự, và điều đó hoàn toàn dễ hiểu.",
        ],
        "follow": [
            "Mình nghĩ trước hết nên tách rõ nguyên nhân khiến bạn bực nhất, rồi mới tính cách phản hồi.",
            "Nếu bạn muốn, mình có thể giúp bạn sắp xếp lại sự việc theo từng ý để bớt rối và bớt nóng hơn.",
        ],
    },
    "lo_lắng": {
        "open": [
            "Mình cảm nhận được bạn đang bất an và suy nghĩ khá nhiều.",
            "Nghe có vẻ bạn đang lo thật sự, chứ không chỉ là hơi băn khoăn.",
        ],
        "follow": [
            "Mình có thể cùng bạn chia nhỏ vấn đề ra từng bước để cảm giác bớt nặng hơn.",
            "Ta thử xác định việc nào cần xử lý trước trong hôm nay để giảm áp lực ngay từ đầu nhé.",
        ],
    },
    "căng_thẳng": {
        "open": [
            "Mình thấy bạn đang ở trạng thái quá tải và phải gồng khá nhiều.",
            "Nghe như nhịp suy nghĩ của bạn đang rất căng và dồn dập.",
        ],
        "follow": [
            "Điều mình nên làm trước là thu gọn vấn đề về vài việc quan trọng nhất, như vậy sẽ dễ thở hơn.",
            "Mình có thể cùng bạn sắp lại mức ưu tiên để bớt cảm giác bị ngập việc.",
        ],
    },
    "thất_vọng": {
        "open": [
            "Mình hiểu cảm giác hụt hẫng khi kết quả chưa phản ánh đúng nỗ lực của bạn.",
            "Nghe vậy mình thấy bạn đã kỳ vọng và đặt tâm huyết khá nhiều vào chuyện này.",
        ],
        "follow": [
            "Điều đó không có nghĩa là cố gắng của bạn vô ích. Mình có thể cùng bạn nhìn lại điểm nào cần chỉnh để lần sau tốt hơn.",
            "Mình nghĩ mình nên tách phần bạn đã làm tốt và phần cần cải thiện, như vậy sẽ công bằng hơn với chính bạn.",
        ],
    },
    "mệt_mỏi": {
        "open": [
            "Mình thấy năng lượng của bạn đang xuống khá thấp.",
            "Nghe như bạn đã mệt trong một khoảng thời gian chứ không chỉ là thoáng qua.",
        ],
        "follow": [
            "Lúc này mình nghĩ ưu tiên nên là giảm tải trước, rồi mới xử lý phần việc còn lại.",
            "Mình có thể giúp bạn chia việc thành phần nhỏ hơn để đỡ kiệt sức hơn.",
        ],
    },
    "trung_tính": {
        "open": [
            "Mình đang theo dõi câu chuyện của bạn.",
            "Mình đã nhận được thông tin bạn chia sẻ.",
        ],
        "follow": [
            "Bạn kể thêm một chút về cảm xúc hoặc tình huống hiện tại để mình phản hồi sát hơn nhé.",
            "Nếu bạn muốn, mình có thể cùng bạn làm rõ điều đang khiến bạn bận tâm nhất.",
        ],
    },
    "không_xác_định": {
        "open": ["Mình đã nhận được nội dung của bạn."],
        "follow": ["Bạn mô tả kỹ hơn một chút để mình nhận diện cảm xúc chính xác hơn nhé."],
    },
}


@dataclass
class EmotionResult:
    source: str
    label: str
    confidence: float
    details: Dict[str, float]
    rationale: str


@dataclass
class FusionResult:
    label: str
    confidence: float
    weights: Dict[str, float]
    rationale: str


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\-_/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def rolling_history_text(history: List[Dict[str, str]], limit: int = 4) -> str:
    user_turns = [item.get("content", "") for item in history if item.get("role") == "user"]
    return " ".join(user_turns[-limit:]).strip()


def inactive_result(source: str, reason: str) -> EmotionResult:
    return EmotionResult(source, "không_xác_định", 0.0, {}, reason)


def detect_concern(text: str) -> str:
    cleaned = normalize_text(text)
    scores: Dict[str, int] = {}
    for concern, keys in CONCERN_RULES.items():
        scores[concern] = sum(1 for k in keys if k in cleaned)
    concern, score = max(scores.items(), key=lambda item: item[1], default=("chung", 0))
    return concern if score > 0 else "chung"


def detect_text_emotion(text: str, history_context: str = "") -> EmotionResult:
    cleaned = normalize_text(text)
    context = normalize_text(history_context)
    if not cleaned:
        return inactive_result("text", "Không có nội dung văn bản.")

    scores = {label: 0.0 for label in EMOTION_LABELS if label not in {"không_xác_định", "trung_tính"}}
    tokens = cleaned.split()

    for emotion, keywords in TEXT_EMOTION_LEXICON.items():
        for kw in keywords:
            if " " in kw:
                if kw in cleaned:
                    scores[emotion] += 1.5
            else:
                for idx, token in enumerate(tokens):
                    if token == kw:
                        boost = 1.0
                        prev_window = set(tokens[max(0, idx - 2): idx])
                        if prev_window & INTENSIFIERS:
                            boost += 0.5
                        if prev_window & NEGATIONS:
                            boost -= 0.8
                        scores[emotion] += max(0.0, boost)

    if any(k in cleaned for k in ["không biết", "không chắc", "sợ không kịp", "áp lực"]):
        scores["lo_lắng"] += 1.0
    if any(k in cleaned for k in ["quá nhiều việc", "ngộp", "rối", "không xoay được"]):
        scores["căng_thẳng"] += 1.0
    if any(k in cleaned for k in ["cố gắng nhiều", "bị phê bình", "bị chê"]):
        scores["thất_vọng"] += 1.0

    if context:
        if any(k in context for k in ["deadline", "thi", "luận văn", "áp lực"]):
            scores["lo_lắng"] += 0.4
            scores["căng_thẳng"] += 0.3
        if any(k in context for k in ["mệt", "mất ngủ", "kiệt sức"]):
            scores["mệt_mỏi"] += 0.3

    scores["căng_thẳng"] += scores["lo_lắng"] * 0.20
    scores["thất_vọng"] += scores["buồn"] * 0.15

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    positive_total = sum(max(v, 0.0) for v in scores.values())
    if best_score <= 0.2:
        return EmotionResult("text", "trung_tính", 0.56, scores, "Văn bản không chứa đủ tín hiệu cảm xúc nổi trội.")

    confidence = min(0.95, 0.58 + (best_score / max(1.0, positive_total)) * 0.35)
    matched_terms = []
    for term in TEXT_EMOTION_LEXICON.get(best_label, []):
        if term in cleaned:
            matched_terms.append(term)
        if len(matched_terms) >= 3:
            break
    rationale = (
        f"Phát hiện nhãn {EMOTION_ALIASES.get(best_label, best_label)} từ các tín hiệu từ khóa"
        + (f": {', '.join(matched_terms)}." if matched_terms else ".")
    )
    return EmotionResult("text", best_label, round(confidence, 2), scores, rationale)


def detect_face_emotion(image: Optional[np.ndarray]) -> EmotionResult:
    if image is None:
        return inactive_result("face", "Không có ảnh khuôn mặt.")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return inactive_result("face", "Không phát hiện được khuôn mặt rõ ràng.")

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    roi = gray[y:y + h, x:x + w]
    smiles = SMILE_CASCADE.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    eyes = EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=7, minSize=(18, 18))

    brightness = float(np.mean(roi)) / 255.0
    contrast = float(np.std(roi)) / 255.0
    details = {
        "smiles": float(len(smiles)),
        "eyes": float(len(eyes)),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
    }

    if len(smiles) > 0:
        label = "vui"
        confidence = 0.77
        rationale = "Phát hiện nụ cười trên vùng miệng nên khuôn mặt nghiêng về trạng thái vui vẻ."
    elif len(eyes) == 0 and brightness < 0.45:
        label = "mệt_mỏi"
        confidence = 0.64
        rationale = "Độ sáng thấp và mắt kém rõ nên khuôn mặt nghiêng về trạng thái mệt mỏi."
    elif brightness < 0.40:
        label = "buồn"
        confidence = 0.58
        rationale = "Biểu cảm trầm, độ sáng thấp và thiếu tín hiệu tích cực nên nghiêng về buồn."
    else:
        label = "trung_tính"
        confidence = 0.60
        rationale = "Khuôn mặt không có đủ tín hiệu mạnh để kết luận cảm xúc nổi bật."

    return EmotionResult("face", label, round(confidence, 2), details, rationale)


def safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def detect_audio_emotion(audio_path: Optional[str]) -> EmotionResult:
    if not audio_path:
        return inactive_result("audio", "Không có tệp âm thanh.")

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return inactive_result("audio", "Âm thanh rỗng hoặc không đọc được.")

        y = librosa.util.normalize(y)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)[0])

        try:
            f0 = librosa.yin(y, fmin=70, fmax=350, sr=sr)
            voiced = f0[np.isfinite(f0)]
            pitch_mean = safe_float(np.mean(voiced)) if voiced.size else 0.0
            pitch_std = safe_float(np.std(voiced)) if voiced.size else 0.0
            voiced_ratio = safe_float(len(voiced) / max(1, len(f0)))
        except Exception:
            pitch_mean = 0.0
            pitch_std = 0.0
            voiced_ratio = 0.0

        rms_mean = safe_float(np.mean(rms))
        rms_std = safe_float(np.std(rms))
        zcr_mean = safe_float(np.mean(zcr))
        centroid_mean = safe_float(np.mean(centroid) / 4000.0)
        duration = safe_float(librosa.get_duration(y=y, sr=sr))

        scores = {label: 0.0 for label in EMOTION_LABELS if label != "không_xác_định"}

        if rms_mean > 0.12 and zcr_mean > 0.08 and centroid_mean > 0.48:
            scores["giận"] += 1.8
            scores["căng_thẳng"] += 1.0
        if rms_mean > 0.10 and pitch_std > 45:
            scores["lo_lắng"] += 1.5
        if rms_mean < 0.06 and centroid_mean < 0.38:
            scores["buồn"] += 1.5
            scores["mệt_mỏi"] += 1.0
        if rms_mean < 0.05 and tempo < 75:
            scores["mệt_mỏi"] += 1.6
        if 0.06 <= rms_mean <= 0.12 and 0.38 <= centroid_mean <= 0.58 and pitch_mean > 170:
            scores["vui"] += 1.4
        if voiced_ratio < 0.35 and duration < 1.2:
            scores["trung_tính"] += 0.8
        if tempo > 120 and pitch_std > 35:
            scores["căng_thẳng"] += 1.0
            scores["lo_lắng"] += 0.6

        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score <= 0.2:
            return EmotionResult(
                "audio",
                "trung_tính",
                0.55,
                {
                    "rms": round(rms_mean, 3),
                    "zcr": round(zcr_mean, 3),
                    "centroid": round(centroid_mean, 3),
                    "tempo": round(tempo, 1),
                    "pitch_mean": round(pitch_mean, 1),
                    "pitch_std": round(pitch_std, 1),
                    "duration": round(duration, 2),
                },
                "Âm thanh chưa có đủ đặc trưng rõ rệt để suy ra một cảm xúc nổi bật.",
            )

        confidence = min(0.90, 0.56 + best_score * 0.12)
        rationale = (
            "Cảm xúc giọng nói được suy ra từ cường độ, nhịp điệu và cao độ. "
            f"RMS={rms_mean:.3f}, pitch_std={pitch_std:.1f}, tempo={tempo:.1f}."
        )
        return EmotionResult(
            "audio",
            best_label,
            round(confidence, 2),
            {
                "rms": round(rms_mean, 3),
                "rms_std": round(rms_std, 3),
                "zcr": round(zcr_mean, 3),
                "centroid": round(centroid_mean, 3),
                "tempo": round(tempo, 1),
                "pitch_mean": round(pitch_mean, 1),
                "pitch_std": round(pitch_std, 1),
                "voiced_ratio": round(voiced_ratio, 3),
                "duration": round(duration, 2),
            },
            rationale,
        )
    except Exception as exc:
        return inactive_result("audio", f"Không đọc được âm thanh: {exc}")


def normalize_selected_sources(
    selected_sources: Optional[Sequence[str]],
    message: str,
    image: Optional[np.ndarray],
    audio_path: Optional[str],
) -> List[str]:
    selected_keys = [SOURCE_DISPLAY_TO_KEY[s] for s in (selected_sources or []) if s in SOURCE_DISPLAY_TO_KEY]
    if selected_keys:
        return selected_keys

    inferred = []
    if normalize_text(message or ""):
        inferred.append("text")
    if image is not None:
        inferred.append("face")
    if audio_path:
        inferred.append("audio")
    return inferred


def fuse_selected_emotions(results: List[EmotionResult], selected_keys: List[str]) -> FusionResult:
    base_weights = {
        "text": 0.45,
        "audio": 0.35,
        "face": 0.20,
    }
    active_results = [r for r in results if r.source in selected_keys and r.label != "không_xác_định" and r.confidence > 0]

    if not selected_keys:
        return FusionResult("không_xác_định", 0.0, {}, "Chưa chọn nguồn dữ liệu và cũng chưa có dữ liệu đầu vào.")

    if not active_results:
        used_labels = ", ".join(SOURCE_LABELS[s] for s in selected_keys)
        return FusionResult("không_xác_định", 0.0, {}, f"Đã chọn {used_labels} nhưng chưa đủ dữ liệu hợp lệ để phân tích.")

    if len(active_results) == 1:
        only = active_results[0]
        return FusionResult(
            only.label,
            only.confidence,
            {only.source: 1.0},
            f"Chế độ đơn nguồn: kết quả cuối lấy trực tiếp từ {SOURCE_LABELS[only.source].lower()}.",
        )

    active_weight_sum = sum(base_weights[r.source] for r in active_results)
    normalized_weights = {r.source: base_weights[r.source] / active_weight_sum for r in active_results}

    label_scores = {label: 0.0 for label in EMOTION_LABELS}
    contributions = []
    for result in active_results:
        src_weight = normalized_weights.get(result.source, 0.0)
        label_scores[result.label] += src_weight * result.confidence
        contributions.append(
            f"{SOURCE_LABELS[result.source]}→{EMOTION_ALIASES.get(result.label, result.label)} ({src_weight:.0%})"
        )
        if result.label == "lo_lắng":
            label_scores["căng_thẳng"] += src_weight * result.confidence * 0.12
        if result.label == "buồn":
            label_scores["thất_vọng"] += src_weight * result.confidence * 0.08
        if result.label == "mệt_mỏi":
            label_scores["căng_thẳng"] += src_weight * result.confidence * 0.06

    final_label, final_score = max(label_scores.items(), key=lambda item: item[1])
    confidence = min(0.96, 0.52 + final_score)
    rationale = "Hợp nhất linh hoạt từ các nguồn đang chọn: " + "; ".join(contributions) + "."
    return FusionResult(final_label, round(confidence, 2), normalized_weights, rationale)


def dominant_recent_emotions(history: List[Dict[str, str]], limit: int = 6) -> List[str]:
    emotions = [item.get("emotion") for item in history if item.get("role") == "assistant" and item.get("emotion")]
    recent = emotions[-limit:]
    return [emotion for emotion, _ in Counter(recent).most_common(2)]


def format_selected_sources(selected_keys: List[str]) -> str:
    if not selected_keys:
        return "Chưa chọn nguồn"
    return " + ".join(SOURCE_LABELS[key] for key in selected_keys)


def source_case_note(selected_keys: List[str], user_text: str) -> str:
    has_text = "text" in selected_keys and normalize_text(user_text or "")
    has_face = "face" in selected_keys
    has_audio = "audio" in selected_keys

    if has_text and has_face and has_audio:
        return "Gợi ý đang dựa trên cả văn bản, khuôn mặt và âm thanh."
    if has_text and has_face:
        return "Gợi ý đang dựa trên văn bản kết hợp biểu cảm khuôn mặt."
    if has_text and has_audio:
        return "Gợi ý đang dựa trên văn bản kết hợp giọng nói."
    if has_face and has_audio and not has_text:
        return "Gợi ý đang dựa trên khuôn mặt kết hợp giọng nói."
    if has_text:
        return "Gợi ý đang dựa chủ yếu trên nội dung văn bản."
    if has_face:
        return "Gợi ý đang dựa chủ yếu trên biểu cảm khuôn mặt."
    if has_audio:
        return "Gợi ý đang dựa chủ yếu trên giọng nói."
    return "Gợi ý đang dựa trên dữ liệu đầu vào hiện có."


def build_chat_suggestions(emotion: str, concern: str, selected_keys: List[str], user_text: str) -> List[str]:
    base = {
        "vui": [
            "Nghe bạn chia sẻ, mình thấy bạn đang khá vui và tích cực.",
            "Đó là tín hiệu rất tốt. Bạn muốn giữ đà này cho việc tiếp theo không?",
            "Bạn có thể kể thêm điều gì làm bạn vui nhất hôm nay không?",
        ],
        "buồn": [
            "Mình thấy lúc này bạn đang buồn và có phần chùng xuống.",
            "Mình sẵn sàng nghe tiếp. Điều gì đang làm bạn nặng lòng nhất?",
            "Bạn muốn mình giúp bạn gỡ từng phần của chuyện này không?",
        ],
        "giận": [
            "Mình thấy bạn đang khá bực và khó chịu.",
            "Ta thử nói rõ điều khiến bạn khó chịu nhất trước nhé.",
            "Mình có thể giúp bạn nghĩ cách phản hồi bình tĩnh hơn.",
        ],
        "lo_lắng": [
            "Mình cảm nhận bạn đang lo lắng khá rõ.",
            "Ta thử tách việc này thành từng bước nhỏ để đỡ áp lực nhé.",
            "Việc gì đang làm bạn lo nhất ở thời điểm này?",
        ],
        "căng_thẳng": [
            "Mình thấy bạn đang căng thẳng và có vẻ bị quá tải.",
            "Ta nên ưu tiên lại việc nào cần làm ngay trước.",
            "Mình có thể giúp bạn chia nhỏ khối việc hiện tại.",
        ],
        "thất_vọng": [
            "Mình thấy bạn đang thất vọng vì kết quả chưa như mong muốn.",
            "Điều đó không có nghĩa là cố gắng của bạn vô ích.",
            "Mình có thể cùng bạn nhìn lại phần nào cần chỉnh sửa.",
        ],
        "mệt_mỏi": [
            "Mình thấy bạn đang mệt và năng lượng xuống khá thấp.",
            "Có lẽ lúc này nên giảm tải một chút trước đã.",
            "Bạn muốn mình giúp chọn việc tối thiểu cần làm hôm nay không?",
        ],
        "trung_tính": [
            "Hiện tại cảm xúc chưa nổi bật rõ, khá gần với trạng thái trung tính.",
            "Bạn có thể chia sẻ thêm để mình phản hồi sát hơn.",
            "Mình đang theo dõi nội dung bạn đưa vào.",
        ],
        "không_xác_định": [
            "Hiện tại hệ thống chưa đủ tín hiệu để kết luận cảm xúc rõ ràng.",
            "Bạn có thể nhập thêm văn bản hoặc bổ sung ảnh, âm thanh.",
            "Mình sẽ phân tích lại ngay khi có thêm dữ liệu.",
        ],
    }[emotion]

    if concern == "học tập":
        base[1] = "Mình gợi ý bạn chốt ngay việc học hoặc luận văn nào cần làm trước."
    elif concern == "công việc":
        base[1] = "Mình gợi ý bạn tách việc gấp, việc quan trọng và việc có thể lùi."
    elif concern == "quan hệ":
        base[1] = "Mình gợi ý bạn nói chậm lại và làm rõ cảm xúc trước khi phản hồi người khác."
    elif concern == "sức khỏe":
        base[1] = "Mình gợi ý bạn ưu tiên nghỉ và giảm tải trước khi ép mình cố thêm."

    note = source_case_note(selected_keys, user_text)
    return [note] + base


def build_response(user_text: str, final_result: FusionResult, history: List[Dict[str, str]], selected_keys: List[str]) -> str:
    emotion = final_result.label if final_result.label in RESPONSE_BANK else "trung_tính"
    concern = detect_concern(user_text)
    suggestions = build_chat_suggestions(emotion, concern, selected_keys, user_text)
    emotion_alias = EMOTION_ALIASES.get(emotion, emotion)
    mode_text = format_selected_sources(selected_keys)

    lines = [
        f"Kết quả cảm xúc: {EMOJI.get(emotion, '⚪')} {emotion_alias}",
        f"Nguồn sử dụng: {mode_text}",
        "Gợi ý đoạn chat:",
    ]
    lines.extend([f"- {item}" for item in suggestions])
    return "\n".join(lines)

def compute_empathy_score(final_emotion: str, response: str, concern: str) -> Tuple[float, Dict[str, float]]:
    response_l = normalize_text(response)
    emotion_markers = {
        "vui": ["mừng", "tích cực", "đáng ghi nhận", "đáng trân trọng"],
        "buồn": ["hiểu", "lắng nghe", "không cần tự gồng", "nặng lòng"],
        "giận": ["khó chịu", "bực", "tách rõ", "bình tĩnh"],
        "lo_lắng": ["chia nhỏ", "ưu tiên", "bất an", "áp lực"],
        "căng_thẳng": ["quá tải", "gồng", "mức ưu tiên", "ngập việc"],
        "thất_vọng": ["nỗ lực", "không vô ích", "công bằng", "tâm huyết"],
        "mệt_mỏi": ["giảm tải", "xuống sức", "kiệt sức", "phần nhỏ"],
        "trung_tính": ["theo dõi", "nói thêm", "hỗ trợ"],
        "không_xác_định": ["mô tả", "rõ hơn"],
    }
    emotion_hits = sum(1 for kw in emotion_markers.get(final_emotion, []) if kw in response_l)
    warmth_hits = sum(1 for kw in ["mình", "cùng", "lắng nghe", "hỗ trợ", "hiểu"] if kw in response_l)
    concern_bonus = 0.08 if concern != "chung" and concern in response_l else 0.04 if concern != "chung" else 0.0

    emotion_alignment = min(1.0, 0.55 + emotion_hits * 0.12)
    warmth = min(1.0, 0.60 + warmth_hits * 0.08)
    relevance = min(1.0, 0.62 + concern_bonus)

    score = min(0.95, 0.45 * emotion_alignment + 0.30 * warmth + 0.25 * relevance)
    return round(score, 2), {
        "emotion_alignment": round(emotion_alignment, 2),
        "warmth": round(warmth, 2),
        "relevance": round(relevance, 2),
    }


def format_emotion_line(title: str, result: EmotionResult | FusionResult) -> str:
    label = result.label
    alias = EMOTION_ALIASES.get(label, label)
    emoji = EMOJI.get(label, "⚪")
    return f"{title}: {emoji} {alias} (độ tin cậy: {result.confidence:.0%})"


def format_detail_block(
    selected_keys: List[str],
    text_result: EmotionResult,
    face_result: EmotionResult,
    audio_result: EmotionResult,
    final_result: FusionResult,
    empathy_score: float,
    empathy_parts: Dict[str, float],
    concern: str,
) -> str:
    lines = [
        f"Chế độ đang chọn: {format_selected_sources(selected_keys)}",
        format_emotion_line("Văn bản", text_result),
        format_emotion_line("Khuôn mặt", face_result),
        format_emotion_line("Âm thanh", audio_result),
        format_emotion_line("Tổng hợp", final_result),
        f"Mối bận tâm chính: {concern}",
        f"Empathy score: {empathy_score:.0%}",
        f"- Emotion alignment: {empathy_parts['emotion_alignment']:.0%}",
        f"- Warmth: {empathy_parts['warmth']:.0%}",
        f"- Relevance: {empathy_parts['relevance']:.0%}",
    ]
    return "\n".join(lines)


def format_rationale_block(
    selected_keys: List[str],
    text_result: EmotionResult,
    face_result: EmotionResult,
    audio_result: EmotionResult,
    final_result: FusionResult,
) -> str:
    selected_text = format_selected_sources(selected_keys)
    return (
        f"[Chế độ] {selected_text}\n\n"
        f"[Văn bản] {text_result.rationale}\n\n"
        f"[Khuôn mặt] {face_result.rationale}\n\n"
        f"[Âm thanh] {audio_result.rationale}\n\n"
        f"[Hợp nhất] {final_result.rationale}"
    )


def chatbot_turn(
    selected_sources: Optional[List[str]],
    message: str,
    image: Optional[np.ndarray],
    audio_path: Optional[str],
    history: Optional[List[Dict[str, str]]],
):
    history = history or []
    history_context = rolling_history_text(history)
    selected_keys = normalize_selected_sources(selected_sources, message or "", image, audio_path)

    if not selected_keys:
        empty_history = history.copy()
        warning = "Hãy chọn ít nhất một nguồn hoặc cung cấp dữ liệu đầu vào để hệ thống tự nhận diện nguồn."
        if message:
            empty_history.append({"role": "user", "content": message})
        empty_history.append({"role": "assistant", "content": warning, "emotion": "không_xác_định"})
        display_history = [{"role": item["role"], "content": item.get("content", "")} for item in empty_history]
        return display_history, empty_history, warning, warning, format_selected_sources(selected_keys), None

    text_result = detect_text_emotion(message or "", history_context) if "text" in selected_keys else inactive_result("text", "Kênh văn bản không được chọn.")
    face_result = detect_face_emotion(image) if "face" in selected_keys else inactive_result("face", "Kênh khuôn mặt không được chọn.")
    audio_result = detect_audio_emotion(audio_path) if "audio" in selected_keys else inactive_result("audio", "Kênh âm thanh không được chọn.")

    final_result = fuse_selected_emotions([text_result, face_result, audio_result], selected_keys)
    concern_text = (message or "") + " " + history_context
    concern = detect_concern(concern_text)
    reply = build_response(message or "", final_result, history, selected_keys)
    empathy_score, empathy_parts = compute_empathy_score(final_result.label, reply, concern)

    user_display = (message or "").strip()
    if not user_display:
        user_display = f"[Không nhập văn bản | Chạy chế độ: {format_selected_sources(selected_keys)}]"

    history.append({
        "role": "user",
        "content": user_display,
        "sources": selected_keys,
    })
    history.append({
        "role": "assistant",
        "content": reply,
        "emotion": final_result.label,
        "sources": selected_keys,
    })

    detail_block = format_detail_block(selected_keys, text_result, face_result, audio_result, final_result, empathy_score, empathy_parts, concern)
    rationale_block = format_rationale_block(selected_keys, text_result, face_result, audio_result, final_result)
    mode_summary = f"Chế độ hiện tại: {format_selected_sources(selected_keys)}"

    display_history = []
    for item in history:
        if item.get("role") in {"user", "assistant"}:
            display_history.append({"role": item["role"], "content": item.get("content", "")})

    return display_history, history, detail_block, rationale_block, mode_summary, None


def clear_chat():
    return [], [], "Đã xóa hội thoại. Bạn có thể bắt đầu lại.", "", "Chưa chọn chế độ", None, None


CSS = """
#detail-box textarea, #rationale-box textarea {
    min-height: 220px !important;
}
#mode-box textarea {
    min-height: 70px !important;
}
footer { display: none !important; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Empathy AI Flexible Demo") as demo:
        gr.Markdown(
            "# Empathy AI\n"
            "Bản demo."
        )

        state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Hội thoại", height=500)
                message = gr.Textbox(
                    label="Nội dung người dùng (tùy chọn nếu chỉ dùng ảnh/âm thanh)",
                    placeholder="",
                    lines=4,
                )
                with gr.Row():
                    send_btn = gr.Button("Phân tích và phản hồi", variant="primary")
                    clear_btn = gr.Button("Xóa hội thoại")
            with gr.Column(scale=2):
                selected_sources = gr.CheckboxGroup(
                    choices=["Văn bản", "Khuôn mặt", "Âm thanh"],
                    value=["Văn bản"],
                    label="Chọn nguồn dữ liệu cần dùng",
                    info="",
                )
                image = gr.Image(
                    sources=["webcam", "upload"],
                    type="numpy",
                    label="Ảnh khuôn mặt (tùy chọn)",
                    height=220,
                )
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Âm thanh / giọng nói (tùy chọn)",
                )
                mode_box = gr.Textbox(
                    label="Tóm tắt chế độ",
                    elem_id="mode-box",
                    interactive=False,
                    value="Chế độ hiện tại: Văn bản",
                )

        with gr.Row():
            details = gr.Textbox(
                label="Kết quả cảm xúc và gợi ý chat",
                elem_id="detail-box",
                lines=11,
                interactive=False,
                value="Chưa có dữ liệu. Kết quả sẽ hiển thị tình trạng cảm xúc cuối cùng và gợi ý đoạn chat theo nguồn đã chọn.",
            )
            rationale = gr.Textbox(
                label="Giải thích mô hình",
                elem_id="rationale-box",
                lines=11,
                interactive=False,
                value="Hệ thống sẽ giải thích riêng cho từng nguồn và phần hợp nhất linh hoạt.",
            )

        send_btn.click(
            fn=chatbot_turn,
            inputs=[selected_sources, message, image, audio, state],
            outputs=[chatbot, state, details, rationale, mode_box, message],
        )
        message.submit(
            fn=chatbot_turn,
            inputs=[selected_sources, message, image, audio, state],
            outputs=[chatbot, state, details, rationale, mode_box, message],
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, state, details, rationale, mode_box, image, audio],
        )

        gr.Markdown(
            
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7862, theme=gr.themes.Soft(), css=CSS)
