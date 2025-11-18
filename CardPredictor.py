# prediction_p1.py
import json, re, time, logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Set

logger = logging.getLogger(__name__)

HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]

# ---------- JSON ----------
def jload(file: str, is_set=False, scalar=False):
    try:
        with open(file) as f:
            data = json.load(f)
            if is_set: return set(data)
            if scalar: return (data.get("active", False) if file == "inter_mode_status.json" else float(data))
            if file == "sequential_history.json": return {int(k): v for k, v in data.items()}
            return data
    except (FileNotFoundError, ValueError):
        if is_set: return set()
        if scalar: return (False if file == "inter_mode_status.json" else 0.0)
        return [] if "inter_data" in file else ({})
    except Exception as e:
        logger.error("jload %s : %s", file, e)
        return set() if is_set else (False if file == "inter_mode_status.json" else ([] if "inter_data" in file else {}))

def jsave(data, file: str):
    out = list(data) if isinstance(data, set) else data
    with open(file, "w") as f: json.dump(out, f, indent=2)

# ---------- CARTES ----------
def extract_game_number(msg: str) -> Optional[int]:
    m = re.search(r'#N(\d+)\.', msg, re.I) or re.search(r'🔵(\d+)🔵', msg)
    return int(m.group(1)) if m else None

def extract_total_points(msg: str) -> Optional[int]:
    m = re.search(r'#T(\d+)', msg)
    return int(m.group(1)) if m else None

def extract_first_parentheses(msg: str) -> Optional[str]:
    m = re.search(r'\(([^)]*)\)', msg)
    return m.group(1).strip() if m else None

def card_details(content: str) -> List[Tuple[str, str]]:
    content = content.replace("❤️", "♥️")
    return re.findall(r'(\d+|[AKQJ])(♠️|♥️|♦️|♣️)', content, re.I)

def first_two_cards(content: str) -> List[str]:
    return [f"{v}{c}" for v, c in card_details(content)[:2]]

def q_in_first_paren(msg: str) -> bool:
    content = extract_first_parentheses(msg)
    if not content: return False
    return any(v.upper() == "Q" for v, _ in card_details(content))

# ---------- INTER ----------
class Predictor:
    def __init__(self):
        self.pred     : Dict[int, Dict] = jload("predictions.json")
        self.proc     : Set[int]        = jload("processed.json", is_set=True)
        self.last     : float           = jload("last_prediction_time.json", scalar=True)
        self.seq_hist : Dict[int, Dict] = jload("sequential_history.json")
        self.inter    : List[Dict]      = jload("inter_data.json")
        self.active   : bool            = jload("inter_mode_status.json", scalar=True)
        self.rules    : List[Dict]      = jload("smart_rules.json")
        self.cooldown = 30

    # ---------- SAVE ----------
    def save_all(self):
        for attr, file in [
            (self.pred, "predictions.json"),
            (self.proc, "processed.json"),
            (self.last, "last_prediction_time.json"),
            (self.seq_hist, "sequential_history.json"),
            (self.inter, "inter_data.json"),
            (self.active, "inter_mode_status.json"),
            (self.rules, "smart_rules.json"),
        ]: jsave(attr, file)

    # ---------- INTER COLLECT ----------
    def collect(self, game: int, msg: str):
        content = extract_first_parentheses(msg)
        if not content: return
        f2 = first_two_cards(content)
        if len(f2) == 2:
            self.seq_hist[game] = {"cards": f2, "date": datetime.now().isoformat()}
        if q_in_first_paren(msg):
            n2 = game - 2
            trig = self.seq_hist.get(n2)
            if trig and not any(e.get("numero_resultat") == game for e in self.inter):
                self.inter.append({
                    "numero_resultat": game,
                    "declencheur": trig["cards"],
                    "numero_declencheur": n2,
                    "carte_q": "Q",
                    "date_resultat": datetime.now().isoformat(),
                })
                self.save_all()

    # ---------- SMART ----------
    def analyze(self):
        counts: Dict[tuple, int] = {}
        for e in self.inter:
            counts[tuple(e["declencheur"])] = counts.get(tuple(e["declencheur"]), 0) + 1
        top3 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        self.rules = [{"cards": list(k), "count": v} for k, v in top3]
        self.active = bool(self.rules)
        self.save_all()

    # ---------- COUNT ----------
    def count_absence_q(self) -> int:
        c = 0
        for gn in sorted(self.seq_hist.keys(), reverse=True):
            if not any(crd.startswith("Q") for crd in self.seq_hist[gn]["cards"]):
                c += 1
            else:
                break
        return c

    # ---------- COOLDOWN ----------
    def can_predict(self) -> bool:
        return time.time() > (self.last + self.cooldown)

    # ---------- INDICATORS ----------
    def pending(self, msg: str) -> bool:
        return "🕐" in msg or "⏰" in msg

    def completed(self, msg: str) -> bool:
        return "✅" in msg or "🔰" in msg

    # ---------- SHOULD ----------
    def should_predict(self, msg: str) -> Tuple[bool, Optional[int], Optional[str]]:
        game = extract_game_number(msg)
        if not game: return False, None, None
        self.collect(game, msg)

        # 1️⃣ Garde-fou
        if self.pending(msg) or not self.completed(msg):
            logger.info("Message non finalisé → aucune règle")
            return False, None, None

        g1_content = extract_first_parentheses(msg)
        if not g1_content: return False, None, None
        g1_vals = [v.upper() for v, _ in card_details(g1_content)]
        all_paren = re.findall(r'\(([^)]*)\)', msg)
        g2_vals = [v.upper() for v, _ in card_details(all_paren[1])] if len(all_paren) > 1 else []

        confidence = ""
        has_j_only = g1_vals.count("J") == 1 and not any(h in g1_vals for h in ("A", "Q", "K"))
        two_j = g1_vals.count("J") >= 2
        high_t = (extract_total_points(msg) or 0) > 40
        three_miss = self.count_absence_q() >= 3

        predicted = None

        # INTER
        if self.active and self.rules:
            trigger = first_two_cards(g1_content)
            if any(tuple(r["cards"]) == tuple(trigger) for r in self.rules):
                predicted = "Q"

        # STATIQUE
        if not predicted:
            if has_j_only:
                predicted, confidence = "Q", "98%"
            elif two_j:
                predicted, confidence = "Q", "57%"
            elif high_t:
                predicted, confidence = "Q", "97%"
            elif three_miss:
                predicted, confidence = "Q", "60%"
            elif {"8", "9", "10"}.issubset(g1_vals) or {"8", "9", "10"}.issubset(g2_vals):
                predicted, confidence = "Q", "70%"
            elif "Q" in g1_vals:
                logger.info("Q déjà dans G1 → pas de préd")
            else:
                # K+J, Tag O/R, double G1 faible
                if ("K" in g1_vals and "J" in g1_vals) or \
                   (re.search(r'\b[OR]\b', msg)) or \
                   (not any(h in g1_vals for h in HIGH_VALUE_CARDS) and
                    not any(h in g2_vals for h in HIGH_VALUE_CARDS) and
                    self.seq_hist.get(game - 1) and
                    not any(h in [re.match(r'(\d+|[AKQJ])', c).group(1)
                                  for c in self.seq_hist[game - 1]['cartes']]
                            for h in HIGH_VALUE_CARDS)):
                    predicted, confidence = "Q", "70%"

        if predicted and not self.can_predict():
            logger.warning("Cooldown")
            return False, None, None

        if predicted:
            h = hash(msg)
            if h not in self.proc:
                self.proc.add(h)
                self.last = time.time()
                self.save_all()
                return True, game, self.make_prediction(game, predicted, confidence)
        return False, None, None

    # ---------- MAKE ----------
    def make_prediction(self, game: int, value: str, confidence: str) -> str:
        target = game + 2
        text = f"🔵{target}🔵:Valeur Q statut :⏳" + (f" {confidence}" if confidence else "")
        self.pred[target] = {
            "predicted_costume": value,
            "status": "pending",
            "predicted_from": game,
            "verification_count": 0,
            "message_text": text,
            "message_id": None,
            "confidence": confidence,
        }
        self.save_all()
        return text

    # ---------- VERIFY ----------
    def verify(self, msg: str) -> Optional[Dict]:
        game = extract_game_number(msg)
        if not game or not self.pred: return None
        for pred_game, pred in self.pred.items():
            if pred.get("status") != "pending" or pred.get("predicted_costume") != "Q":
                continue
            offset = game - pred_game
            if 0 <= offset <= 2:
                symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                q_found = q_in_first_paren(msg)
                if q_found:
                    symbol = symbol_map[offset]
                    new_text = f"🔵{pred_game}🔵:Valeur Q statut :{symbol}"
                    pred["status"] = f"correct_offset_{offset}"
                    pred["final_message"] = new_text
                    self.save_all()
                    logger.info("Verification SUCCÈS +%s N=%s", offset, game)
                    return {"type": "edit_message", "predicted_game": pred_game, "new_message": new_text}
                if offset == 2 and not q_found:
                    new_text = f"🔵{pred_game}🔵:Valeur Q statut :❌"
                    pred["status"] = "failed"
                    pred["final_message"] = new_text
                    self.save_all()
                    logger.info("Verification ÉCHEC +2 N=%s", game)
                    return {"type": "edit_message", "predicted_game": pred_game, "new_message": new_text}
        return None
        # prediction_p2.py
import os, logging, requests
from flask import Flask, request, abort
from prediction_p1 import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
SOURCE_ID = int(os.getenv("SOURCE_ID") or 0)
PRED_ID   = int(os.getenv("PRED_ID") or 0)

app = Flask(__name__)
pred = Predictor()

# ---------- TELEGRAM ----------
def send_text(chat_id: int, text: str) -> int:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    return r.json()["result"]["message_id"]

def edit_text(chat_id: int, msg_id: int, text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    requests.post(url, json={"chat_id": chat_id, "message_id": msg_id, "text": text}, timeout=10)

# ---------- WEBHOOK ----------
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)
    if not data: abort(400)

    # message canal SOURCE
    msg = data.get("channel_post") or data.get("edited_channel_post")
    if not msg: abort(400)
    if msg.get("chat", {}).get("id") != SOURCE_ID: abort(400)

    text = msg.get("text") or msg.get("caption") or ""
    do_it, game, pred_text = pred.should_predict(text)

    if do_it and pred_text:
        mid = send_text(PRED_ID, pred_text)
        pred.pred[game + 2]["message_id"] = mid
        pred.save_all()

    # vérification (edit)
    result = pred.verify(text)
    if result and result["type"] == "edit_message":
        pg = result["predicted_game"]
        mid = pred.pred[pg]["message_id"]
        if mid:
            edit_text(PRED_ID, mid, result["new_message"])

    return "", 200

# ---------- MAIN ----------
if __name__ == "__main__":
    # gunicorn entry-point : gunicorn prediction_p2:app
    logging.info("Webhook ready – /webhook listening")
                
