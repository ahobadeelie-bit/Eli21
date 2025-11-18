# card_predictor_p1.py
# Partie 1 / 2 : CardPredictor – logique complète (prédiction + vérification)
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]


class CardPredictor:
    """Prédiction & vérification Q – finalisé & sécurisé."""

    def __init__(self):
        # --- Persistance ---
        self.predictions: Dict[int, Dict]          = self._load("predictions.json")
        self.processed: Set[int]                   = self._load("processed.json", is_set=True)
        self.last_pred_time: float                 = self._load("last_prediction_time.json", scalar=True)
        self.sequential_history: Dict[int, Dict]   = self._load("sequential_history.json")
        self.inter_data: List[Dict]                = self._load("inter_data.json")
        self.is_inter_active: bool                 = self._load("inter_mode_status.json", scalar=True)
        self.smart_rules: List[Dict]               = self._load("smart_rules.json")

        self.prediction_cooldown = 30

        if self.inter_data and not self.is_inter_active:
            self.analyze_and_set_smart_rules(initial_load=True)

    # ---------- JSON ----------
    def _load(self, file: str, is_set=False, scalar=False):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                if is_set: return set(data)
                if scalar:
                    if file == "inter_mode_status.json": return data.get("active", False)
                    return float(data) if isinstance(data, (int, float)) else 0.0
                if file == "sequential_history.json":
                    return {int(k): v for k, v in data.items()}
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            if is_set: return set()
            if scalar: return (False if file == "inter_mode_status.json" else 0.0)
            return [] if "inter_data" in file else ({})
        except Exception as e:
            logger.error("Load %s : %s", file, e)
            return set() if is_set else (False if file == "inter_mode_status.json" else ([] if "inter_data" in file else {}))

    def _save(self, data, file: str):
        out = list(data) if isinstance(data, set) else data
        try:
            with open(file, "w") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            logger.error("Save %s : %s", file, e)

    def save_all(self):
        for attr, fname in [
            (self.predictions, "predictions.json"),
            (self.processed, "processed.json"),
            (self.last_pred_time, "last_prediction_time.json"),
            (self.sequential_history, "sequential_history.json"),
            (self.inter_data, "inter_data.json"),
            (self.is_inter_active, "inter_mode_status.json"),
            (self.smart_rules, "smart_rules.json"),
        ]:
            self._save(attr, fname)

    # ---------- EXTRACTION ----------
    def extract_game_number(self, msg: str) -> Optional[int]:
        m = re.search(r'#N(\d+)\.', msg, re.I) or re.search(r'🔵(\d+)🔵', msg)
        return int(m.group(1)) if m else None

    def extract_total_points(self, msg: str) -> Optional[int]:
        m = re.search(r'#T(\d+)', msg)
        return int(m.group(1)) if m else None

    def extract_first_parentheses_content(self, msg: str) -> Optional[str]:
        m = re.search(r'\(([^)]*)\)', msg)
        return m.group(1).strip() if m else None

    def extract_card_details(self, content: str) -> List[Tuple[str, str]]:
        content = content.replace("❤️", "♥️")
        return re.findall(r'(\d+|[AKQJ])(♠️|♥️|♦️|♣️)', content, re.I)

    def get_first_two_cards(self, content: str) -> List[str]:
        details = self.extract_card_details(content)[:2]
        return [f"{v}{c}" for v, c in details]

    def check_q_in_first_parentheses(self, msg: str) -> Optional[Tuple[str, str]]:
        content = self.extract_first_parentheses_content(msg)
        if not content:
            return None
        for val, col in self.extract_card_details(content):
            if val.upper() == "Q":
                logger.info("Q détectée : Q%s", col)
                return val, col
        return None

    # ---------- INTER ----------
    def collect_inter_data(self, game: int, msg: str):
        content = self.extract_first_parentheses_content(msg)
        if not content:
            return
        first_two = self.get_first_two_cards(content)
        if len(first_two) == 2:
            self.sequential_history[game] = {"cartes": first_two, "date": datetime.now().isoformat()}
        if self.check_q_in_first_parentheses(msg):
            n2 = game - 2
            trigger = self.sequential_history.get(n2)
            if trigger:
                if any(e.get("numero_resultat") == game for e in self.inter_data):
                    logger.warning("Doublon INTER ignoré N=%s", game)
                    return
                self.inter_data.append(
                    {
                        "numero_resultat": game,
                        "declencheur": trigger["cartes"],
                        "numero_declencheur": n2,
                        "carte_q": "Q",
                        "date_resultat": datetime.now().isoformat(),
                    }
                )
                self.save_all()

    def analyze_and_set_smart_rules(self, initial_load=False):
        counts: Dict[tuple, int] = {}
        for e in self.inter_data:
            counts[tuple(e["declencheur"])] = counts.get(tuple(e["declencheur"]), 0) + 1
        top3 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        self.smart_rules = [{"cards": list(k), "count": v} for k, v in top3]
        self.is_inter_active = bool(self.smart_rules) or initial_load
        self.save_all()

    # ---------- CONFIDENCE ----------
    def count_absence_q(self) -> int:
        c = 0
        for gn in sorted(self.sequential_history.keys(), reverse=True):
            if not any(crd.startswith("Q") for crd in self.sequential_history[gn]["cartes"]):
                c += 1
            else:
                break
        return c

    # ---------- COOLDOWN ----------
    def can_predict(self) -> bool:
        return time.time() > (self.last_pred_time + self.prediction_cooldown)

    # ---------- INDICATEURS ----------
    def has_pending_indicators(self, msg: str) -> bool:
        return "🕐" in msg or "⏰" in msg

    def has_completion_indicators(self, msg: str) -> bool:
        return "✅" in msg or "🔰" in msg

    # ---------- SHOULD_PREDICT ----------
    def should_predict(self, message: str) -> Tuple[bool, Optional[int], Optional[str]]:
        if not self.target_channel_id:
            return False, None, None

        game = self.extract_game_number(message)
        if not game:
            return False, None, None

        # Collecte INTER systématique
        self.collect_inter_data(game, message)

        # 1️⃣ Garde-fou : message non finalisé
        if self.has_pending_indicators(message):
            return False, None, None
        if not self.has_completion_indicators(message):
            logger.info("❌ PRÉDICTION BLOQUÉE: ✅/🔰 absents OU ⏰ présent")
            return False, None, None

        # 2️⃣ Extraction groupes
        g1_content = self.extract_first_parentheses_content(message)
        if not g1_content:
            return False, None, None

        g1_cards = self.extract_card_details(g1_content)
        g1_vals = [v.upper() for v, _ in g1_cards]

        # G2 (second parenthèse)
        all_paren = re.findall(r'\(([^)]*)\)', message)
        g2_vals = [v.upper() for v, _ in self.extract_card_details(all_paren[1])] if len(all_paren) > 1 else []

        # 3️⃣ Confiance & prédiction
        confidence = ""
        has_j_only = g1_vals.count("J") == 1 and not any(h in g1_vals for h in ("A", "Q", "K"))
        two_j = g1_vals.count("J") >= 2
        high_tension = (self.extract_total_points(message) or 0) > 40
        three_missing = self.count_absence_q() >= 3

        predicted = None

        # Mode INTER prioritaire
        if self.is_inter_active and self.smart_rules:
            trigger = self.get_first_two_cards(g1_content)
            if any(tuple(rule["cards"]) == tuple(trigger) for rule in self.smart_rules):
                predicted = "Q"
                logger.info("🔮 PRÉDICTION INTER: déclencheur %s", trigger)

        # Règles statiques
        if not predicted:
            if has_j_only:
                predicted = "Q"
                confidence = "98%"
            elif two_j:
                predicted = "Q"
                confidence = "57%"
            elif high_tension:
                predicted = "Q"
                confidence = "97%"
            elif three_missing:
                predicted = "Q"
                confidence = "60%"
            elif {"8", "9", "10"}.issubset(g1_vals) or {"8", "9", "10"}.issubset(g2_vals):
                predicted = "Q"
                confidence = "70%"
            elif "Q" in g1_vals:
                logger.info("🔮 Q déjà dans G1 → pas de préd")
                predicted = None
            else:
                # K+J, Tag O/R, double G1 faible, etc.
                if ("K" in g1_vals and "J" in g1_vals) or \
                   (re.search(r'\b[OR]\b', message)) or \
                   (not any(h in g1_vals for h in HIGH_VALUE_CARDS) and
                    not any(h in g2_vals for h in HIGH_VALUE_CARDS) and
                    self.sequential_history.get(game - 1) and
                    not any(h in [re.match(r'(\d+|[AKQJ])', c).group(1)
                                  for c in self.sequential_history[game - 1]['cartes']]
                            for h in HIGH_VALUE_CARDS)):
                    predicted = "Q"
                    confidence = "70%"

        if predicted and not self.can_predict():
            logger.warning("⏳ Cooldown actif")
            return False, None, None

        if predicted:
            h = hash(message)
            if h not in self.processed:
                self.processed.add(h)
                self.last_pred_time = time.time()
                self.save_all()
                return True, game, self.make_prediction(game, predicted, confidence)

        return False, None, None

    # ---------- MAKE PREDICTION ----------
    def make_prediction(self, game: int, value: str, confidence: str) -> str:
        target = game + 2
        text = f"🔵{target}🔵:Valeur Q statut :⏳" + (f" {confidence}" if confidence else "")
        self.predictions[target] = {
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

    # ---------- VERIFICATION ----------
    def verify_prediction(self, text: str, is_edited: bool = False) -> Optional[Dict]:
        game = self.extract_game_number(text)
        if not game or not self.predictions:
            return None

        for pred_game, pred in self.predictions.items():
            if pred.get("status") != "pending" or pred.get("predicted_costume") != "Q":
                continue
            offset = game - pred_game
            if 0 <= offset <= 2:
                symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                q_found = self.check_q_in_first_parentheses(text)
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
      # card_predictor_p2.py
# Partie 2 / 2 : handlers Telegram + constantes + mini-main
import os
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from card_predictor_p1 import CardPredictor  # import classe

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------- TOKEN ----------
BOT_TOKEN = os.getenv("BOT_TOKEN") or "PASTE_YOUR_TOKEN_HERE"

# ---------- BOT ----------
predictor = CardPredictor()


# ---------- HANDLERS ----------
async def handle_new_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg or msg.chat_id != predictor.target_channel_id:
        return
    text = msg.text or msg.caption or ""
    do_it, game, pred_text = predictor.should_predict(text)
    if do_it and pred_text:
        sent = await context.bot.send_message(
            chat_id=predictor.prediction_channel_id, text=pred_text
        )
        # on stock l'ID pour edition future
        predictor.predictions[game + 2]["message_id"] = sent.message_id
        predictor.save_all()


async def handle_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.edited_channel_post
    if not msg or msg.chat_id != predictor.target_channel_id:
        return
    text = msg.text or msg.caption or ""
    result = predictor.verify_prediction(text, is_edited=True)
    if result and result["type"] == "edit_message":
        msg_id = predictor.predictions[result["predicted_game"]]["message_id"]
        if msg_id:
            await context.bot.edit_message_text(
                chat_id=predictor.prediction_channel_id,
                message_id=msg_id,
                text=result["new_message"],
            )


# ---------- MAIN ----------
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edited_message), group=1)
    logger.info("Bot démarré – prêt à prédire Q")
    app.run_polling()


if __name__ == "__main__":
    main()
  
