
"""
=============================================================================
PONG FROM PIXELS
=============================================================================

WHAT IS THIS?
-------------
This program teaches a computer to play the video game Pong by itself!
It learns just by watching the screen (pixels) and getting points.

HOW DOES IT LEARN?
------------------
1. The computer looks at the game screen
2. It decides: should I move my paddle UP or DOWN?
3. It plays the game and sees if it scored or lost a point
4. It remembers what worked and what didn't
5. Over time, it gets better and better!


Generative AI assistance used in the development

=============================================================================
"""

# STEP 1: IMPORT THE TOOLS WE NEED
# =============================================================================

import numpy as np
import pickle
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import matplotlib.pyplot as plt

print("✓ All tools loaded successfully!")

# STEP 2: SETTINGS
# =============================================================================

BRAIN_SIZE = 200
LEARNING_SPEED = 0.001
FUTURE_IMPORTANCE = 0.99
GAMES_PER_UPDATE = 10
SHOW_GAME = False       # Set True to watch (slower training)
TOTAL_GAMES = 1000

print("✓ Settings configured!")



# STEP 3: HELPER FUNCTIONS
# =============================================================================

def squish_number(x):
    """Squishes any number to be between 0 and 1"""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def clean_up_image(game_screen):
    """Simplify the game screen from 100,800 numbers to 6,400"""
    game_screen = game_screen[35:195]
    game_screen = game_screen[::2, ::2, 0]
    game_screen[game_screen == 144] = 0
    game_screen[game_screen == 109] = 0
    game_screen[game_screen != 0] = 1
    return game_screen.astype(float).flatten()


def calculate_rewards(rewards_list):
    """Figure out how good each move was"""
    final_rewards = np.zeros(len(rewards_list))
    running_total = 0
    for t in range(len(rewards_list) - 1, -1, -1):
        if rewards_list[t] != 0:
            running_total = 0
        running_total = running_total * FUTURE_IMPORTANCE + rewards_list[t]
        final_rewards[t] = running_total
    return final_rewards



# STEP 4: THE BRAIN (Neural Network)
# =============================================================================

def create_brain():
    """Creates a fresh brain for our AI"""
    brain = {
        'weights1': np.random.randn(BRAIN_SIZE, 6400) / np.sqrt(6400),
        'weights2': np.random.randn(BRAIN_SIZE) / np.sqrt(BRAIN_SIZE)
    }
    print("✓ New brain created!")
    print(f"  Connections: {brain['weights1'].size + brain['weights2'].size:,}")
    return brain


def think(brain, what_i_see):
    """Brain decides: UP or DOWN?"""
    hidden = np.dot(brain['weights1'], what_i_see)
    hidden = np.maximum(hidden, hidden * 0.01)
    output = np.dot(brain['weights2'], hidden)
    probability_up = squish_number(output)
    return probability_up, hidden


def learn_from_game(brain, game_data):
    """Update brain based on what worked"""
    screens = game_data['screens']
    hiddens = game_data['hiddens']
    actions = game_data['actions']
    rewards = game_data['rewards']
    
    processed_rewards = calculate_rewards(rewards)
    processed_rewards -= np.mean(processed_rewards)
    if np.std(processed_rewards) > 0:
        processed_rewards /= np.std(processed_rewards)
    
    gradients1 = np.zeros_like(brain['weights1'])
    gradients2 = np.zeros_like(brain['weights2'])
    
    for i in range(len(screens)):
        action_taken = 1 if actions[i] == 2 else 0
        prob_up, _ = think(brain, screens[i])
        gradient = (action_taken - prob_up) * processed_rewards[i]
        gradients2 += gradient * hiddens[i]
        hidden_gradient = gradient * brain['weights2']
        hidden_gradient[hiddens[i] <= 0] *= 0.01
        gradients1 += np.outer(hidden_gradient, screens[i])
    
    brain['weights1'] += LEARNING_SPEED * gradients1
    brain['weights2'] += LEARNING_SPEED * gradients2
    return brain



# STEP 5: SAVE AND LOAD
# =============================================================================

def save_brain(brain, game_number, filename='pong_brain.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'brain': brain, 'game': game_number}, f)
    print(f"💾 Brain saved at game {game_number}!")


def load_brain(filename='pong_brain.pkl'):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 Brain loaded from game {data['game']}!")
        return data['brain'], data['game']
    except FileNotFoundError:
        print("📂 No saved brain found, starting fresh!")
        return None, 0



# STEP 6: TRAINING
# =============================================================================

def train():
    """Train the AI to play Pong!"""
    
    print("\n" + "="*60)
    print("🏓 PONG AI TRAINING STARTING!")
    print("="*60 + "\n")
    
    brain = create_brain()
    
    # Create game - using simpler method
    if SHOW_GAME:
        game = gym.make('ALE/Pong-v5', render_mode='human', disable_env_checker=True)
    else:
        game = gym.make('ALE/Pong-v5', disable_env_checker=True)
    
    all_scores = []
    average_score = -21
    
    screen, _ = game.reset()
    previous_screen = None
    
    game_data = {'screens': [], 'hiddens': [], 'actions': [], 'rewards': []}
    current_score = 0
    games_played = 0
    
    print("🎮 Starting training...")
    print("   (This will take a while - Pong is hard!)\n")
    
    while games_played < TOTAL_GAMES:
        current_clean = clean_up_image(screen)
        
        if previous_screen is None:
            screen_diff = np.zeros(6400)
        else:
            screen_diff = current_clean - previous_screen
        previous_screen = current_clean
        
        prob_up, hidden = think(brain, screen_diff)
        
        if np.random.random() < prob_up:
            action = 2  # UP
        else:
            action = 3  # DOWN
        
        game_data['screens'].append(screen_diff)
        game_data['hiddens'].append(hidden)
        game_data['actions'].append(action)
        
        screen, reward, done, truncated, _ = game.step(action)
        game_data['rewards'].append(reward)
        current_score += reward
        
        if done or truncated:
            games_played += 1
            
            if games_played % GAMES_PER_UPDATE == 0:
                brain = learn_from_game(brain, game_data)
            
            all_scores.append(current_score)
            average_score = 0.95 * average_score + 0.05 * current_score
            
            print(f"Game {games_played:4d} | Score: {current_score:+3.0f} | "
                  f"Average: {average_score:+6.2f}")
            
            if games_played % 100 == 0:
                save_brain(brain, games_played)
            
            screen, _ = game.reset()
            previous_screen = None
            game_data = {'screens': [], 'hiddens': [], 'actions': [], 'rewards': []}
            current_score = 0
    
    save_brain(brain, games_played, 'pong_brain_final.pkl')
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Final average score: {average_score:+.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_scores, alpha=0.5)
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title('Pong AI Learning Progress')
    plt.savefig('training_progress.png')
    print("📊 Graph saved to 'training_progress.png'")
    
    game.close()
    return brain



# STEP 7: WATCH THE AI PLAY
# =============================================================================

def watch_ai_play(num_games=3):
    """Load a trained brain and watch it play!"""
    
    brain, _ = load_brain('pong_brain_final.pkl')
    if brain is None:
        brain, _ = load_brain('pong_brain.pkl')
    if brain is None:
        print("❌ No trained brain found! Run training first.")
        return
    
    game = gym.make('ALE/Pong-v5', render_mode='human', disable_env_checker=True)
    
    for i in range(num_games):
        print(f"\n🎮 Playing game {i+1}...")
        screen, _ = game.reset()
        previous_screen = None
        score = 0
        done = False
        
        while not done:
            current_clean = clean_up_image(screen)
            if previous_screen is None:
                screen_diff = np.zeros(6400)
            else:
                screen_diff = current_clean - previous_screen
            previous_screen = current_clean
            
            prob_up, _ = think(brain, screen_diff)
            action = 2 if prob_up > 0.5 else 3
            
            screen, reward, done, truncated, _ = game.step(action)
            done = done or truncated
            score += reward
        
        print(f"   Final score: {score:+.0f}")
    
    game.close()


# STEP 8: RUN THE PROGRAM
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🏓"*30 + "\n")
    print("PONG FROM PIXELS")
    print("Teaching a computer to play Pong!")
    print("\n" + "🏓"*30 + "\n")
    
    print("What would you like to do?")
    print("  1. Train a new AI")
    print("  2. Watch a trained AI play")
    print("  3. Quick test")
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        watch_ai_play()
    elif choice == "3":
        print("\n🔧 Running quick test...")
        brain = create_brain()
        test_screen = np.random.randn(6400)
        prob, _ = think(brain, test_screen)
        print(f"✓ Brain works! Output: {prob:.4f}")
        
        # Also test game creation
        print("✓ Testing game...")
        game = gym.make('ALE/Pong-v5', disable_env_checker=True)
        screen, _ = game.reset()
        print(f"✓ Game works! Screen shape: {screen.shape}")
        game.close()
        print("✓ All systems go!")
    else:
        print("Please enter 1, 2, or 3")