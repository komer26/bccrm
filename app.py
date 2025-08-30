from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import csv
from datetime import datetime, timedelta
import requests


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-please-change")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "participants.csv"
CONFIG_PATH = DATA_DIR / "config.json"
BRACKET_PATH = DATA_DIR / "bracket.json"
STANDINGS_PATH = DATA_DIR / "standings.json"
CSV_HEADERS = [
    "timestamp",
    "last_name",
    "first_name",
    "photo_filename",
    "role",
    "phone",
    "phone_verified_at",
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Rktdth1993")
BACKGROUND_VIDEO_URL = os.environ.get("BACKGROUND_VIDEO_URL", "")
BACKGROUND_VIDEO_URL_WEBM = os.environ.get("BACKGROUND_VIDEO_URL_WEBM", "")
BACKGROUND_POSTER_URL = os.environ.get("BACKGROUND_POSTER_URL", "")
BACKGROUND_VIDEO_FILE = os.environ.get("BACKGROUND_VIDEO_FILE", "video.mp4")
SMS_DEBUG = os.environ.get("SMS_DEBUG", "1") in {"1", "true", "True", "yes"}
EVENT_START_ISO = os.environ.get("EVENT_START_ISO", "2025-08-30T10:00:00")
SMSC_LOGIN = os.environ.get("SMSC_LOGIN", "")
SMSC_PASSWORD = os.environ.get("SMSC_PASSWORD", "")
SMSC_SENDER = os.environ.get("SMSC_SENDER", "")  # опционально, буквенное имя
VERIFY_CODE_TTL_SECONDS = int(os.environ.get("VERIFY_CODE_TTL_SECONDS", "600"))
VERIFY_CODE_COOLDOWN_SECONDS = int(os.environ.get("VERIFY_CODE_COOLDOWN_SECONDS", "60"))
VERIFY_MAX_ATTEMPTS = int(os.environ.get("VERIFY_MAX_ATTEMPTS", "5"))


def _ensure_storage_ready() -> None:
    """Создает необходимые директории и CSV-файл с заголовком при первом запуске."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    else:
        _migrate_csv_schema()
    # базовый конфиг
    if not CONFIG_PATH.exists():
        try:
            _write_config({"sms_verification_enabled": True})
        except Exception:
            pass


def _migrate_csv_schema() -> None:
    """Миграция CSV: приводит заголовок к CSV_HEADERS, заполняя недостающие поля пустыми."""
    try:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if not first_row:
                # Пустой файл — просто запишем корректный заголовок
                with CSV_PATH.open("w", newline="", encoding="utf-8") as wf:
                    writer = csv.writer(wf)
                    writer.writerow(CSV_HEADERS)
                return
            header = first_row
    except FileNotFoundError:
        return

    if header != CSV_HEADERS:
        # Перечитываем все строки как dict по старому формату и перезаписываем с новой схемой
        rows = []
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        _write_rows(rows)


def _read_config() -> dict:
    try:
        import json
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"sms_verification_enabled": True}
        data.setdefault("sms_verification_enabled", True)
        return data
    except Exception:
        return {"sms_verification_enabled": True}


def _write_config(cfg: dict) -> None:
    import json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _generate_code(length: int = 6) -> str:
    import random
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def _send_sms(phone: str, message: str) -> None:
    """Отправка SMS через smsc.ru или dev-flash."""
    phone_norm = "+" + "".join(ch for ch in phone if ch.isdigit()) if not phone.startswith("+") else phone
    if SMS_DEBUG or not (SMSC_LOGIN and SMSC_PASSWORD):
        flash(f"Код подтверждения (dev): {message}")
        return
    try:
        resp = requests.get(
            "https://smsc.ru/sys/send.php",
            params={
                "login": SMSC_LOGIN,
                "psw": SMSC_PASSWORD,
                "phones": phone_norm,
                "mes": f"Код: {message}",
                "fmt": 3,
                **({"sender": SMSC_SENDER} if SMSC_SENDER else {}),
            },
            timeout=10,
        )
        data = resp.json()
        if "id" in data or data.get("cnt"):
            flash("Код отправлен по SMS.")
        else:
            flash("Не удалось отправить SMS (SMSC). Проверьте номер/баланс/лимиты.")
    except Exception:
        flash("Ошибка связи с SMSC. Код показан (dev) в сообщении.")
        if SMS_DEBUG:
            flash(f"Код подтверждения (dev): {message}")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _seconds_since(iso_str: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_str)
        return int((datetime.now() - dt).total_seconds())
    except Exception:
        return 10**9


def _is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def register_form():
    _ensure_storage_ready()
    cfg = _read_config()
    return render_template("register.html", SMS_VERIFICATION_ENABLED=bool(cfg.get("sms_verification_enabled", True)))
@app.get("/verify")
def verify_phone():
    pending = session.get("pending_registration")
    if not pending:
        flash("Нет данных для подтверждения. Пожалуйста, заполните форму заново.")
        return redirect(url_for("register_form"))
    cfg = _read_config()
    if not bool(cfg.get("sms_verification_enabled", True)):
        flash("Подтверждение по SMS отключено.")
        return redirect(url_for("register_form"))
    ttl_left = max(0, VERIFY_CODE_TTL_SECONDS - _seconds_since(pending.get("created_at", "")))
    cooldown_left = max(0, VERIFY_CODE_COOLDOWN_SECONDS - _seconds_since(pending.get("last_sent_at", pending.get("created_at", ""))))
    return render_template("verify.html", phone=pending.get("phone"), ttl_left=ttl_left, cooldown_left=cooldown_left)


@app.post("/verify")
def verify_phone_post():
    pending = session.get("pending_registration")
    if not pending:
        flash("Сессия подтверждения истекла. Пожалуйста, заполните форму заново.")
        return redirect(url_for("register_form"))
    cfg = _read_config()
    if not bool(cfg.get("sms_verification_enabled", True)):
        flash("Подтверждение по SMS отключено.")
        return redirect(url_for("register_form"))
    code_input = request.form.get("code", "").strip()
    if not code_input:
        flash("Введите код подтверждения.")
        return redirect(url_for("verify_phone"))
    # TTL
    if _seconds_since(pending.get("created_at", "")) > VERIFY_CODE_TTL_SECONDS:
        flash("Срок действия кода истёк. Отправьте код ещё раз.")
        return redirect(url_for("verify_phone"))
    # Лимит попыток
    attempts = int(pending.get("attempts", 0))
    if attempts >= VERIFY_MAX_ATTEMPTS:
        flash("Слишком много попыток. Отправьте код ещё раз.")
        return redirect(url_for("verify_phone"))
    if code_input != pending.get("code"):
        pending["attempts"] = attempts + 1
        session["pending_registration"] = pending
        flash("Неверный код. Проверьте SMS и попробуйте снова.")
        return redirect(url_for("verify_phone"))

    # Подтверждено — записываем участника в CSV
    created_at = _now_iso()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            created_at,
            pending.get("last_name", ""),
            pending.get("first_name", ""),
            pending.get("photo_filename", ""),
            "",  # role
            pending.get("phone", ""),
            created_at,  # phone_verified_at
        ])

    session.pop("pending_registration", None)
    flash("Телефон подтверждён! Регистрация завершена.")
    return redirect(url_for("participants_list"))


@app.post("/verify/resend")
def verify_phone_resend():
    pending = session.get("pending_registration")
    if not pending:
        flash("Сессия подтверждения истекла. Пожалуйста, заполните форму заново.")
        return redirect(url_for("register_form"))
    # Cooldown между отправками
    since_last = _seconds_since(pending.get("last_sent_at", pending.get("created_at", "")))
    if since_last < VERIFY_CODE_COOLDOWN_SECONDS:
        wait = VERIFY_CODE_COOLDOWN_SECONDS - since_last
        flash(f"Повторная отправка будет доступна через {wait} сек.")
        return redirect(url_for("verify_phone"))

    # Новый код и сброс попыток
    new_code = _generate_code()
    pending["code"] = new_code
    pending["created_at"] = _now_iso()
    pending["last_sent_at"] = _now_iso()
    pending["attempts"] = 0
    pending["resends"] = int(pending.get("resends", 0)) + 1
    session["pending_registration"] = pending

    try:
        _send_sms(pending.get("phone", ""), new_code)
    except Exception:
        flash("Не удалось отправить SMS. Попробуйте позже.")
    else:
        flash("Код повторно отправлен.")
    return redirect(url_for("verify_phone"))


@app.post("/register")
def handle_register():
    _ensure_storage_ready()

    last_name = request.form.get("last_name", "").strip()
    first_name = request.form.get("first_name", "").strip()
    phone = request.form.get("phone", "").strip()
    photo = request.files.get("photo")
    cfg = _read_config()

    if not last_name or not first_name:
        flash("Введите, пожалуйста, фамилию и имя.")
        return redirect(url_for("register_form"))

    if photo is None or photo.filename == "":
        flash("Пожалуйста, прикрепите фотографию.")
        return redirect(url_for("register_form"))

    if not _is_allowed_file(photo.filename):
        flash("Допустимые форматы фото: jpg, jpeg, png, gif, webp.")
        return redirect(url_for("register_form"))

    # Проверку телефона выполняем только если включено SMS-подтверждение
    if bool(cfg.get("sms_verification_enabled", True)):
        digits_only = "".join(ch for ch in phone if ch.isdigit())
        if len(digits_only) < 10:
            flash("Введите корректный номер телефона (не менее 10 цифр).")
            return redirect(url_for("register_form"))
    else:
        # Если SMS отключено — телефон не обязателен
        phone = phone or ""

    safe_original = secure_filename(photo.filename)
    timestamp_fs = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = secure_filename(f"{last_name}_{first_name}")
    filename = f"{timestamp_fs}_{base_name}_{safe_original}"
    filepath = UPLOAD_DIR / filename
    photo.save(filepath)

    cfg = _read_config()
    if bool(cfg.get("sms_verification_enabled", True)):
        # Подготовка подтверждения номера: генерируем код, сохраняем данные в сессии
        code = _generate_code()
        session["pending_registration"] = {
            "last_name": last_name,
            "first_name": first_name,
            "phone": phone,
            "photo_filename": filename,
            "code": code,
            "created_at": _now_iso(),
            "last_sent_at": _now_iso(),
            "attempts": 0,
            "resends": 0,
        }
        try:
            _send_sms(phone, code)
        except Exception:
            # В случае ошибки отправки всё равно позволим подтвердить код (в dev-режиме он во flash)
            flash("Не удалось отправить SMS, попробуйте ещё раз или свяжитесь с организатором.")
        return redirect(url_for("verify_phone"))
    else:
        # SMS подтверждение отключено — пишем сразу
        created_at = _now_iso()
        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                created_at,
                last_name,
                first_name,
                filename,
                "",  # role
                phone,
                created_at,  # считаем подтверждённым
            ])
        flash("Регистрация отправлена без подтверждения SMS.")
        return redirect(url_for("participants_list"))


@app.get("/list")
def participants_list():
    _ensure_storage_ready()

    participants = []
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role_val = (row.get("role", "") or "").strip().lower()
                if role_val == "org":
                    continue
                participants.append({
                    "timestamp": row.get("timestamp", ""),
                    "last_name": row.get("last_name", ""),
                    "first_name": row.get("first_name", ""),
                    "photo_url": url_for("static", filename=f"uploads/{row.get('photo_filename', '')}"),
                    "role": row.get("role", ""),
                    "phone": row.get("phone", ""),
                    "phone_verified_at": row.get("phone_verified_at", ""),
                })

    participants.sort(key=lambda x: x["timestamp"])  # по времени создания
    return render_template("list.html", participants=participants)


@app.get("/organizers")
def organizers_list():
    _ensure_storage_ready()

    organizers = []
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role_val = (row.get("role", "") or "").strip().lower()
                if role_val != "org":
                    continue
                organizers.append({
                    "timestamp": row.get("timestamp", ""),
                    "last_name": row.get("last_name", ""),
                    "first_name": row.get("first_name", ""),
                    "photo_url": url_for("static", filename=f"uploads/{row.get('photo_filename', '')}"),
                    "role": row.get("role", ""),
                    "phone": row.get("phone", ""),
                    "phone_verified_at": row.get("phone_verified_at", ""),
                })

    organizers.sort(key=lambda x: x["timestamp"])  # по времени создания
    return render_template("organizers.html", organizers=organizers)


@app.context_processor
def inject_globals():
    # Вычисляем URL фонового видео: приоритет внешнему URL,
    # затем файлы в static/, затем fallback в uploads/
    computed_video_url = BACKGROUND_VIDEO_URL
    if not computed_video_url:
        # Приоритет: оптимизированное видео из uploads
        opt_upload = UPLOAD_DIR / "video-optimized.mp4"
        if opt_upload.exists():
            computed_video_url = url_for("static", filename="uploads/video-optimized.mp4")

    if not computed_video_url:
        # Попробуем найти mp4 в static/
        for name in ("background.mp4", "video.mp4", "video-optimized.mp4", BACKGROUND_VIDEO_FILE):
            if not name:
                continue
            p = STATIC_DIR / name
            if p.exists():
                computed_video_url = url_for("static", filename=name)
                break
        # Если не нашли по стандартным именам — возьмём первый попавшийся mp4
        if not computed_video_url:
            for p in STATIC_DIR.glob("*.mp4"):
                computed_video_url = url_for("static", filename=p.name)
                break
    if not computed_video_url:
        # Fallback: uploads/
        local_path = UPLOAD_DIR / BACKGROUND_VIDEO_FILE
        if BACKGROUND_VIDEO_FILE and local_path.exists():
            computed_video_url = url_for("static", filename=f"uploads/{BACKGROUND_VIDEO_FILE}")

    # webm источник: из переменной среды, иначе ищем в static/
    computed_webm_url = BACKGROUND_VIDEO_URL_WEBM
    if not computed_webm_url:
        for name in ("background.webm", "video.webm", "video-optimized.webm"):
            p = STATIC_DIR / name
            if p.exists():
                computed_webm_url = url_for("static", filename=name)
                break
        if not computed_webm_url:
            for p in STATIC_DIR.glob("*.webm"):
                computed_webm_url = url_for("static", filename=p.name)
                break

    # постер для видео
    computed_poster_url = BACKGROUND_POSTER_URL
    if not computed_poster_url:
        for name in ("background.jpg", "background.jpeg", "background.png", "background.webp", "poster.jpg", "poster.webp"):
            p = STATIC_DIR / name
            if p.exists():
                computed_poster_url = url_for("static", filename=name)
                break

    cfg = _read_config()
    return {
        "BACKGROUND_VIDEO_URL": computed_video_url,
        "BACKGROUND_VIDEO_URL_WEBM": computed_webm_url,
        "BACKGROUND_POSTER_URL": computed_poster_url,
        "SMS_VERIFICATION_ENABLED": bool(cfg.get("sms_verification_enabled", True)),
        "EVENT_START_ISO": EVENT_START_ISO,
    }


def _read_rows():
    rows = []
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _write_rows(rows):
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for r in rows:
            writer.writerow([
                r.get("timestamp", ""),
                r.get("last_name", ""),
                r.get("first_name", ""),
                r.get("photo_filename", ""),
                r.get("role", ""),
            ])


def _require_admin():
    if not session.get("is_admin"):
        flash("Требуется вход в админку.")
        return redirect(url_for("admin_login"))
    return None


# --------- Блок: Турнирная сетка ---------
def _read_bracket():
    try:
        import json
        with BRACKET_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_bracket(data: dict) -> None:
    import json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data["updated_at"] = _now_iso()
    except Exception:
        pass
    with BRACKET_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_standings() -> dict | None:
    try:
        import json
        with STANDINGS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_standings(first_id: int | None, second_id: int | None, third_id: int | None) -> None:
    import json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "first_id": first_id,
        "second_id": second_id,
        "third_id": third_id,
        "updated_at": _now_iso(),
    }
    with STANDINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _participants_for_bracket():
    participants = []
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                role_val = (row.get("role", "") or "").strip().lower()
                if role_val == "org":
                    continue
                participants.append({
                    "id": idx,
                    "last_name": row.get("last_name", ""),
                    "first_name": row.get("first_name", ""),
                    "photo_filename": row.get("photo_filename", ""),
                    "name": f"{row.get('last_name','')} {row.get('first_name','')}".strip(),
                })
    return participants


def _build_bracket_from_participants(participants: list[dict], tables: int = 1, seed_indices: list[int] | None = None) -> dict:
    import math
    import random
    # Список индексов участников (0..len-1) в порядке посева
    if seed_indices:
        # нормализуем: только валидные индексы и без дублей, затем добавим недостающих
        seen = set()
        seeds = []
        for s in seed_indices:
            if isinstance(s, int) and 0 <= s < len(participants) and s not in seen:
                seeds.append(s)
                seen.add(s)
        for i in range(len(participants)):
            if i not in seen:
                seeds.append(i)
    else:
        seeds = list(range(len(participants)))
        random.shuffle(seeds)
    # до ближайшей степени двойки
    def next_pow2(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()
    size = next_pow2(len(seeds))
    # добавим None как BYE
    while len(seeds) < size:
        seeds.append(None)

    # Первый раунд: попарно
    rounds: list[list[dict]] = []
    first_round: list[dict] = []
    for i in range(0, size, 2):
        s1 = seeds[i]
        s2 = seeds[i + 1] if i + 1 < size else None
        first_round.append({
            "id": f"r1m{(i//2)+1}",
            "p1_from": {"seed": s1} if s1 is not None else None,
            "p2_from": {"seed": s2} if s2 is not None else None,
            "winner": None,
        })
    rounds.append(first_round)

    # Последующие раунды: победители предыдущего
    current_size = len(first_round)
    round_index = 2
    while current_size > 1:
        prev_round_idx = round_index - 2  # индекс предыдущего в массиве
        new_round: list[dict] = []
        for i in range(0, current_size, 2):
            m1 = i
            m2 = i + 1
            new_round.append({
                "id": f"r{round_index}m{(i//2)+1}",
                "p1_from": {"winner_round": prev_round_idx, "winner_match": m1},
                "p2_from": {"winner_round": prev_round_idx, "winner_match": m2},
                "winner": None,
            })
        rounds.append(new_round)
        current_size = len(new_round)
        round_index += 1

    # назначим столы по кругу для каждого раунда
    try:
        tnum = max(1, int(tables))
    except Exception:
        tnum = 1
    for r in rounds:
        for idx, m in enumerate(r):
            m["table"] = (idx % tnum) + 1
    # Бронзовый матч (за 3-е место): между проигравшими полуфиналов, если они есть
    bronze = None
    if len(rounds) >= 2 and len(rounds[-2]) >= 2:
        bronze = {
            "id": "bronze",
            "p1_from": {"loser_round": len(rounds) - 2, "loser_match": 0},
            "p2_from": {"loser_round": len(rounds) - 2, "loser_match": 1},
            "winner": None,
            "table": 1,
        }

    return {
        "generated_at": _now_iso(),
        "participants": participants,
        "rounds": rounds,
        "tables": tnum,
        "bronze": bronze,
    }


def _resolve_slot(bracket: dict, slot: dict | None):
    if slot is None:
        return None
    # из посева
    if "seed" in slot:
        seed = slot["seed"]
        if seed is None:
            return None
        parts = bracket.get("participants", [])
        if 0 <= seed < len(parts):
            return parts[seed]
        return None
    # из победителя другого матча
    if "winner_round" in slot and "winner_match" in slot:
        r = slot["winner_round"]
        m = slot["winner_match"]
        try:
            match = bracket["rounds"][r][m]
        except Exception:
            return None
        w = match.get("winner")
        if w == 1:
            return _resolve_slot(bracket, match.get("p1_from"))
        if w == 2:
            return _resolve_slot(bracket, match.get("p2_from"))
        return None
    # из проигравшего другого матча (для бронзового)
    if "loser_round" in slot and "loser_match" in slot:
        r = slot["loser_round"]
        m = slot["loser_match"]
        try:
            match = bracket["rounds"][r][m]
        except Exception:
            return None
        w = match.get("winner")
        if w == 1:
            return _resolve_slot(bracket, match.get("p2_from"))
        if w == 2:
            return _resolve_slot(bracket, match.get("p1_from"))
        return None
    return None


def _bracket_viewmodel(bracket: dict) -> dict:
    rounds_vm: list[list[dict]] = []
    for r_index, rnd in enumerate(bracket.get("rounds", [])):
        row: list[dict] = []
        for m_index, match in enumerate(rnd):
            p1 = _resolve_slot(bracket, match.get("p1_from"))
            p2 = _resolve_slot(bracket, match.get("p2_from"))
            row.append({
                "id": match.get("id"),
                "r": r_index,
                "m": m_index,
                "p1": p1,
                "p2": p2,
                "winner": match.get("winner"),
                "table": match.get("table"),
            })
        rounds_vm.append(row)
    bronze = bracket.get("bronze")
    bronze_vm = None
    if bronze:
        bronze_vm = {
            "id": bronze.get("id"),
            "p1": _resolve_slot(bracket, bronze.get("p1_from")),
            "p2": _resolve_slot(bracket, bronze.get("p2_from")),
            "winner": bronze.get("winner"),
            "table": bronze.get("table"),
        }
    return {"rounds": rounds_vm, "participants": bracket.get("participants", []), "tables": bracket.get("tables", 1), "bronze": bronze_vm}


def _bracket_complete(bracket: dict) -> bool:
    # все матчи всех раундов имеют winner 1/2
    for rnd in bracket.get("rounds", []):
        for match in rnd:
            if match.get("winner") not in (1, 2):
                return False
    # если есть бронза — она тоже должна быть определена
    bronze = bracket.get("bronze")
    if bronze is not None:
        if bronze.get("winner") not in (1, 2):
            return False
    return True


def _compute_standings(bracket: dict) -> list[dict]:
    try:
        final_match = bracket["rounds"][-1][0]
    except Exception:
        return []
    p1 = _resolve_slot(bracket, final_match.get("p1_from"))
    p2 = _resolve_slot(bracket, final_match.get("p2_from"))
    w = final_match.get("winner")
    if w not in (1, 2) or not p1 or not p2:
        return []
    first = p1 if w == 1 else p2
    second = p2 if w == 1 else p1
    # третье место — победитель бронзового матча, если он есть
    bronze = bracket.get("bronze")
    third = None
    if bronze and bronze.get("winner") in (1, 2):
        bp1 = _resolve_slot(bracket, bronze.get("p1_from"))
        bp2 = _resolve_slot(bracket, bronze.get("p2_from"))
        if bronze.get("winner") == 1:
            third = bp1
        else:
            third = bp2
    return [first, second] + ([third] if third else [])


def _standings_from_manual(parts: list[dict]) -> list[dict]:
    manual = _read_standings()
    if not manual:
        return []
    id_to_part = {p.get("id"): p for p in parts}
    result: list[dict] = []
    for key in ("first_id", "second_id", "third_id"):
        pid = manual.get(key)
        if pid is None:
            continue
        p = id_to_part.get(pid)
        if p:
            result.append(p)
    return result


@app.get("/bracket")
def public_bracket():
    _ensure_storage_ready()
    bracket = _read_bracket()
    parts = _participants_for_bracket()
    manual_standings = _standings_from_manual(parts)
    if not bracket:
        return render_template(
            "bracket.html",
            view=None,
            UPDATED_AT=None,
            COMPLETE=bool(manual_standings),
            STANDINGS=manual_standings,
        )
    view = _bracket_viewmodel(bracket)
    complete = _bracket_complete(bracket)
    standings = manual_standings if manual_standings else (_compute_standings(bracket) if complete else [])
    return render_template(
        "bracket.html",
        view=view,
        UPDATED_AT=bracket.get("updated_at") or bracket.get("generated_at"),
        COMPLETE=bool(standings),
        STANDINGS=standings,
    )


@app.get("/bracket.json")
def public_bracket_json():
    _ensure_storage_ready()
    bracket = _read_bracket()
    updated = None
    view = None
    complete = False
    standings = []
    parts = _participants_for_bracket()
    manual_standings = _standings_from_manual(parts)
    standings_updated = None
    manual = _read_standings()
    if manual:
        standings_updated = manual.get("updated_at")
    if bracket:
        updated = bracket.get("updated_at") or bracket.get("generated_at")
        view = _bracket_viewmodel(bracket)
        complete = _bracket_complete(bracket)
        standings = manual_standings if manual_standings else (_compute_standings(bracket) if complete else [])
        # обновление — берём максимальную метку: сетка/итоги
        if standings_updated and updated:
            updated = max(str(updated), str(standings_updated))
        elif standings_updated and not updated:
            updated = standings_updated
    else:
        standings = manual_standings
        updated = standings_updated
    resp = jsonify({
        "updated_at": updated,
        "view": view,
        "complete": bool(standings),
        "standings": standings,
    })
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/admin/standings")
def admin_standings():
    if (resp := _require_admin()) is not None:
        return resp
    parts = _participants_for_bracket()
    manual = _read_standings() or {}
    return render_template("admin/standings.html", participants=parts, manual=manual)


@app.post("/admin/standings")
def admin_standings_post():
    if (resp := _require_admin()) is not None:
        return resp
    def to_int(val):
        try:
            return int(val)
        except Exception:
            return None
    first = to_int(request.form.get("first"))
    second = to_int(request.form.get("second"))
    third = to_int(request.form.get("third"))
    # нельзя дублировать
    picks = [p for p in (first, second, third) if p is not None]
    if len(set(picks)) != len(picks):
        flash("Выбраны дублирующиеся участники. Исправьте выбор.")
        return redirect(url_for("admin_standings"))
    _write_standings(first, second, third)
    flash("Итоги сохранены.")
    return redirect(url_for("admin_standings"))


@app.get("/admin/bracket")
def admin_bracket():
    if (resp := _require_admin()) is not None:
        return resp
    bracket = _read_bracket()
    parts = _participants_for_bracket()
    view = _bracket_viewmodel(bracket) if bracket else None
    return render_template("admin/bracket.html", view=view, participants=parts)


@app.post("/admin/bracket/generate")
def admin_bracket_generate():
    if (resp := _require_admin()) is not None:
        return resp
    parts = _participants_for_bracket()
    if len(parts) < 2:
        flash("Недостаточно участников для генерации сетки (нужно минимум 2).")
        return redirect(url_for("admin_bracket"))
    try:
        tables = int(request.form.get("tables", "1"))
    except Exception:
        tables = 1
    if tables < 1:
        tables = 1
    if tables > 32:
        tables = 32
    # если был сохранён ручной посев — используем
    seed_order = session.get("manual_seed_order")
    bracket = _build_bracket_from_participants(parts, tables=tables, seed_indices=seed_order)
    session.pop("manual_seed_order", None)
    _write_bracket(bracket)
    flash("Сетка создана.")
    return redirect(url_for("admin_bracket"))


@app.get("/admin/bracket/seed")
def admin_bracket_seed():
    if (resp := _require_admin()) is not None:
        return resp
    parts = _participants_for_bracket()
    # по умолчанию — текущий порядок
    items = []
    for i, p in enumerate(parts):
        items.append({"idx": i, "name": f"{p['last_name']} {p['first_name']}", "photo": p.get("photo_filename", "")})
    return render_template("admin/seed.html", items=items)


@app.post("/admin/bracket/seed")
def admin_bracket_seed_post():
    if (resp := _require_admin()) is not None:
        return resp
    parts = _participants_for_bracket()
    # ожидаем список индексов idx[] в порядке посева
    indices = request.form.getlist("idx[]") or request.form.getlist("idx")
    try:
        order = [int(x) for x in indices]
    except Exception:
        flash("Некорректные данные посева.")
        return redirect(url_for("admin_bracket_seed"))
    # нормализация произойдет в _build_bracket_from_participants
    session["manual_seed_order"] = order
    flash("Порядок посева сохранён. Сгенерируйте сетку.")
    return redirect(url_for("admin_bracket"))


@app.post("/admin/bracket/reset")
def admin_bracket_reset():
    if (resp := _require_admin()) is not None:
        return resp
    try:
        BRACKET_PATH.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    flash("Сетка сброшена.")
    return redirect(url_for("admin_bracket"))


@app.post("/admin/bracket/winner")
def admin_bracket_winner():
    if (resp := _require_admin()) is not None:
        return resp
    bracket = _read_bracket()
    if not bracket:
        flash("Сетка не найдена.")
        return redirect(url_for("admin_bracket"))
    try:
        r = int(request.form.get("r", ""))
        m = int(request.form.get("m", ""))
        w = int(request.form.get("winner", ""))
        assert w in (1, 2)
    except Exception:
        flash("Некорректные данные.")
        return redirect(url_for("admin_bracket"))
    try:
        match = bracket["rounds"][r][m]
    except Exception:
        flash("Матч не найден.")
        return redirect(url_for("admin_bracket"))
    # Проверим, что есть такой слот
    p1 = _resolve_slot(bracket, match.get("p1_from"))
    p2 = _resolve_slot(bracket, match.get("p2_from"))
    if w == 1 and not p1:
        flash("Нельзя выбрать победителем пустой слот.")
        return redirect(url_for("admin_bracket"))
    if w == 2 and not p2:
        flash("Нельзя выбрать победителем пустой слот.")
        return redirect(url_for("admin_bracket"))
    match["winner"] = w
    _write_bracket(bracket)
    flash("Результат матча сохранён.")
    return redirect(url_for("admin_bracket"))


@app.post("/admin/bracket/winner_bronze")
def admin_bracket_winner_bronze():
    if (resp := _require_admin()) is not None:
        return resp
    bracket = _read_bracket()
    if not bracket or not bracket.get("bronze"):
        flash("Матч за 3-е место не найден.")
        return redirect(url_for("admin_bracket"))
    try:
        w = int(request.form.get("winner", ""))
        assert w in (1, 2)
    except Exception:
        flash("Некорректные данные.")
        return redirect(url_for("admin_bracket"))
    bronze = bracket["bronze"]
    p1 = _resolve_slot(bracket, bronze.get("p1_from"))
    p2 = _resolve_slot(bracket, bronze.get("p2_from"))
    if w == 1 and not p1:
        flash("Нельзя выбрать победителем пустой слот.")
        return redirect(url_for("admin_bracket"))
    if w == 2 and not p2:
        flash("Нельзя выбрать победителем пустой слот.")
        return redirect(url_for("admin_bracket"))
    bronze["winner"] = w
    _write_bracket(bracket)
    flash("Результат матча за 3-е место сохранён.")
    return redirect(url_for("admin_bracket"))


@app.post("/admin/settings/sms")
def admin_settings_sms():
    if (resp := _require_admin()) is not None:
        return resp
    value = request.form.get("enabled", "1").strip()
    cfg = _read_config()
    cfg["sms_verification_enabled"] = (value == "1")
    _write_config(cfg)
    flash("Настройка SMS подтверждения сохранена")
    return redirect(url_for("admin_index"))

@app.get("/admin")
def admin_index():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    rows = _read_rows()
    cfg = _read_config()
    return render_template("admin/index.html", rows=rows, SMS_VERIFICATION_ENABLED=bool(cfg.get("sms_verification_enabled", True)))


@app.get("/admin/login")
def admin_login():
    return render_template("admin/login.html")


@app.post("/admin/login")
def admin_login_post():
    password = request.form.get("password", "")
    if password == ADMIN_PASSWORD:
        session["is_admin"] = True
        flash("Добро пожаловать в админку!")
        return redirect(url_for("admin_index"))
    flash("Неверный пароль.")
    return redirect(url_for("admin_login"))


@app.post("/admin/logout")
def admin_logout():
    session.clear()
    flash("Вы вышли из админки.")
    return redirect(url_for("admin_login"))


@app.get("/admin/edit/<int:idx>")
def admin_edit(idx: int):
    if (resp := _require_admin()) is not None:
        return resp
    rows = _read_rows()
    if idx < 0 or idx >= len(rows):
        flash("Участник не найден")
        return redirect(url_for("admin_index"))
    return render_template("admin/edit.html", idx=idx, row=rows[idx])


@app.post("/admin/edit/<int:idx>")
def admin_edit_post(idx: int):
    if (resp := _require_admin()) is not None:
        return resp
    rows = _read_rows()
    if idx < 0 or idx >= len(rows):
        flash("Участник не найден")
        return redirect(url_for("admin_index"))
    last_name = request.form.get("last_name", "").strip()
    first_name = request.form.get("first_name", "").strip()
    if not last_name or not first_name:
        flash("Заполните фамилию и имя")
        return redirect(url_for("admin_edit", idx=idx))
    role = request.form.get("role", "").strip()
    rows[idx]["last_name"] = last_name
    rows[idx]["first_name"] = first_name
    rows[idx]["role"] = role
    _write_rows(rows)
    flash("Данные обновлены")
    return redirect(url_for("admin_index"))


@app.post("/admin/delete/<int:idx>")
def admin_delete(idx: int):
    if (resp := _require_admin()) is not None:
        return resp
    rows = _read_rows()
    if idx < 0 or idx >= len(rows):
        flash("Участник не найден")
        return redirect(url_for("admin_index"))
    # удалить файл если есть
    photo_filename = rows[idx].get("photo_filename")
    if photo_filename:
        try:
            (UPLOAD_DIR / photo_filename).unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    del rows[idx]
    _write_rows(rows)
    flash("Участник удалён")
    return redirect(url_for("admin_index"))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8448")), debug=True)


