# 🏓 Pong from Pixels

**Teaching a computer to play Pong just by looking at the screen!**

This project uses Reinforcement Learning to train an AI that learns to play Atari Pong from scratch. It starts knowing nothing and gradually learns by trial and error - just like a human would.

---

## 🎯 What Does This Do?

The AI:
1. **Sees** the game screen (raw pixels)
2. **Thinks** using a neural network brain
3. **Decides** to move the paddle UP or DOWN
4. **Learns** from wins and losses

After thousands of games, it becomes a skilled Pong player!

---

## 🚀 Quick Start

### Step 1: Install Dependencies

Open **Anaconda Prompt** and run:

```bash
pip install numpy gymnasium matplotlib
pip install ale-py
pip install autorom
AutoROM --accept-license
```

### Step 2: Run the Program

```bash
python pong_simple.py
```

### Step 3: Choose an Option

```
What would you like to do?
  1. Train a new AI
  2. Watch a trained AI play
  3. Quick test
```

- **Option 3** - Run this first to make sure everything works
- **Option 1** - Start training (takes a few hours)
- **Option 2** - Watch your trained AI play

---

## 🧠 How It Works

### The Brain (Neural Network)

```
Game Screen (6,400 pixels)
         ↓
   Hidden Layer (200 neurons)
         ↓
   Output (probability of moving UP)
```

- **Input**: Simplified 80x80 black & white game image (6,400 numbers)
- **Hidden Layer**: 200 neurons that detect patterns
- **Output**: A number between 0 and 1 (probability of moving UP)

### The Learning Process

1. The AI plays a game and records every move
2. At the end, it sees if it won or lost points
3. Moves that led to scoring get reinforced
4. Moves that led to losing get discouraged
5. Repeat thousands of times!

This is called **Policy Gradient Reinforcement Learning**.

---

## 📊 What to Expect During Training

| Games | Average Score | What's Happening |
|-------|---------------|------------------|
| 0-100 | -21 to -19 | Random flailing, losing badly |
| 100-500 | -19 to -17 | Starting to track the ball |
| 500-1000 | -17 to -14 | Learning basic defense |
| 1000-2000 | -14 to -10 | Getting competitive |
| 2000+ | -10 and up | Actually winning points! |

**Note**: Scores range from -21 (lost every point) to +21 (won every point).

---

## ⚙️ Settings You Can Change

At the top of `pong_simple.py`:

```python
BRAIN_SIZE = 200        # More neurons = smarter but slower
LEARNING_SPEED = 0.001  # Higher = learns faster but less stable
TOTAL_GAMES = 1000      # More games = better player
SHOW_GAME = False       # Set True to watch (much slower)
```

**Tip**: Keep `SHOW_GAME = False` during training. It's 10x faster!

---

## 📁 Files Created

| File | Description |
|------|-------------|
| `pong_brain.pkl` | Saved brain (auto-saves every 100 games) |
| `pong_brain_final.pkl` | Final trained brain |
| `training_progress.png` | Graph showing learning progress |

---

## ❓ FAQ

**Q: Why does it start so bad?**  
A: The brain starts with random connections. It literally knows nothing! Learning takes time.

**Q: How long does training take?**  
A: About 2-4 hours for 1000 games on a typical laptop.

**Q: Can I stop and resume training?**  
A: The brain saves every 100 games. To resume, you'd need to modify the code to load the saved brain.

**Q: Why -21 score at the start?**  
A: Pong is first to 21 points. Score of -21 means the AI lost every single point.

**Q: What's a "good" final score?**  
A: Anything above -10 means it's competitive. Above 0 means it's winning more than losing!

---

## 🔧 Troubleshooting

### "Namespace ALE not found"
Run these commands:
```bash
pip install ale-py
pip install autorom
AutoROM --accept-license
```

### "No module named ale_py"
```bash
pip install ale-py
```

### Game runs but no window appears
Change `SHOW_GAME = True` in the settings, or use option 2 to watch a trained AI.

---

## 📚 Learn More

- [Original Blog Post by Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/) - The inspiration for this project
- [Reinforcement Learning Explained (YouTube)](https://www.youtube.com/watch?v=JgvyzIkgxF0)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## 🎓 Key Concepts

| Term | Simple Explanation |
|------|-------------------|
| **Reinforcement Learning** | Learning by trial and error |
| **Neural Network** | A mathematical "brain" made of connected numbers |
| **Policy Gradient** | Learning which actions lead to rewards |
| **Reward** | +1 for scoring, -1 for opponent scoring |
| **Episode** | One complete game (until someone gets 21 points) |

---
