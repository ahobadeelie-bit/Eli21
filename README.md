# 🤖 Bot de Prédiction DAME (Q) - Telegram

Bot Telegram intelligent pour la prédiction de cartes, spécialement conçu pour anticiper l'apparition de la Dame (Q) dans les tirages en utilisant une stratégie basée sur l'analyse des figures (Valet, Roi, As).

## 🌟 Caractéristiques

- **Mode Webhook** : Optimisé pour un déploiement production sur Replit ou Render.com
- **Stratégie Intelligente** : Analyse des figures (J, K, A) pour prédire l'apparition de la Dame
- **Surveillance de Canal** : Écoute automatique des tirages depuis un canal source
- **Prédictions Automatiques** : Envoi des prédictions vers un canal de prédiction
- **Gestion d'État** : Suivi des échecs et activation automatique du mode intelligent

## 📋 Prérequis

- Python 3.11+
- Token de bot Telegram (via BotFather)
- IDs des canaux Telegram (source et prédiction)

## 🚀 Déploiement Rapide

### Sur Replit

1. Configurer les secrets dans Replit Secrets :
   - `BOT_TOKEN` : Jeton API du bot
   - `ADMIN_CHAT_ID` : Votre ID de chat Telegram
   - `TARGET_CHANNEL_ID` : ID du canal source (format négatif)
   - `PREDICTION_CHANNEL_ID` : ID du canal de prédiction (format négatif)

2. Le bot démarre automatiquement sur le port 5000

3. Configurer le webhook :
   ```bash
   python scripts/setup_webhook.py
   ```

### Sur Render.com

1. Générer le package de déploiement :
   ```bash
   python scripts/deploy.py
   ```

2. Uploader `scripts/bot_telegram_render_*.zip` vers un repo GitHub

3. Créer un Web Service sur Render.com :
   - Connectez votre repo GitHub
   - Type : Web Service
   - Build Command : `pip install -r requirements.txt`
   - Start Command : `gunicorn --bind 0.0.0.0:$PORT --reuse-port main:application`

4. Configurer les variables d'environnement (mêmes que Replit)

5. Après déploiement, appelez `https://votre-app.onrender.com/set_webhook`

## 📁 Structure du Projet

```
.
├── main.py              # Point d'entrée Flask, routes webhook
├── bot.py               # Classe TelegramBot pour l'API
├── handlers.py          # Gestionnaires de commandes et logique
├── card_predictor.py    # Logique de prédiction intelligente
├── config.py            # Configuration et variables d'environnement
├── requirements.txt     # Dépendances Python
├── Procfile            # Configuration pour Render.com
├── render.yaml         # Configuration automatique Render
└── scripts/            # Scripts de test et déploiement
    ├── deploy.py              # Générateur de package Render
    ├── setup_webhook.py       # Configuration webhook automatique
    ├── test_bot.py           # Test du bot
    └── test_channel_prediction.py  # Test prédiction canal
```

## 🎮 Commandes Disponibles

| Commande | Description |
|----------|-------------|
| `/start` | Message de bienvenue |
| `/help` | Affiche la liste des commandes |
| `/status` | État du Mode Intelligent et compteur d'échecs |
| `/inter` | Analyse l'historique et propose l'activation du Mode Intelligent |
| `/defaut` | Désactive le Mode Intelligent |

## 🧠 Mode Intelligent

Le bot utilise une stratégie basée sur la détection de figures pour prédire l'apparition de la Dame :

| Signal Détecté (N-1) | Règle | Jeu Cible | Interprétation |
|---------------------|-------|-----------|----------------|
| **Valet (J) seul** (sans A ni K) | Q_IMMEDIATE | **N+2** | Messager de la Dame |
| **Roi (K) + Valet (J)** | Q_IMMEDIATE | **N+2** | Forte corrélation |
| **Double Valet (J...J)** | Q_IMMEDIATE_JJ | **N+2** | Signal fort et direct |
| **Roi (K) seul** (sans J ni A) | Q_NEXT_DRAW | **N+3** | Domination masculine temporaire |
| **As (A) + Roi (K)** | Q_WAIT_1 | **N+3** | Blocage puis bascule |

### Activation du Mode Intelligent

Le Mode Intelligent peut être activé de deux manières :

1. **Manuellement** via la commande `/inter`
2. **Automatiquement** après 2 échecs consécutifs de prédiction

## 🔧 Configuration

### Variables d'Environnement

| Variable | Description | Exemple |
|----------|-------------|---------|
| `BOT_TOKEN` | Jeton d'API du bot Telegram | `7722770680:AAEblH...` |
| `ADMIN_CHAT_ID` | ID du chat admin pour alertes | `5622847726` |
| `TARGET_CHANNEL_ID` | Canal source (négatif) | `-1003424179389` |
| `PREDICTION_CHANNEL_ID` | Canal prédiction (négatif) | `-1003362820311` |
| `PORT` | Port du serveur (auto sur Replit/Render) | `5000` ou `10000` |

### Obtenir les IDs de Canaux

Pour obtenir l'ID d'un canal :

1. Ajoutez le bot au canal comme administrateur
2. Envoyez un message dans le canal
3. Utilisez l'API Telegram : `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Cherchez le `chat.id` dans la réponse (format négatif)

## 🧪 Tests

```bash
# Tester le bot
python scripts/test_bot.py

# Tester la réception de messages du canal
python scripts/test_channel_prediction.py

# Configurer le webhook manuellement
python scripts/setup_webhook.py
```

## 📊 Workflow de Fonctionnement

1. **Réception** : Le bot écoute les messages du canal source via webhook
2. **Analyse** : Extraction du numéro de jeu et des cartes du premier groupe
3. **Détection** : Identification des figures (J, K, A) si le Mode Intelligent est actif
4. **Prédiction** : Application de la stratégie et calcul du jeu cible (N+2 ou N+3)
5. **Envoi** : Publication de la prédiction dans le canal de prédiction
6. **Vérification** : Validation des prédictions et mise à jour du compteur d'échecs

## 📝 Notes de Développement

- Date de dernière mise à jour : 13 novembre 2025
- Projet importé depuis GitHub et adapté pour Replit
- Mode webhook activé (pas de polling)
- Support complet des messages de canaux publics
- Logs détaillés pour le debugging

## 🛠️ Technologies Utilisées

- **Flask** : Framework web pour les webhooks
- **Gunicorn** : Serveur WSGI de production
- **Requests** : Client HTTP pour l'API Telegram
- **Python 3.11** : Langage de programmation

## 📄 Licence

Ce projet est un bot privé pour usage personnel.

## 🤝 Support

Pour toute question ou problème, contactez l'administrateur via Telegram.
