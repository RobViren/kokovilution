import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple
from kokoro_onnx import Kokoro
import librosa
from scipy.spatial.distance import cosine
import random
import argparse
import os
import gc

@dataclass
class VoiceGenome:
    """Represents a single voice style configuration"""
    embedding: np.ndarray
    fitness: float = 0.0

class VoiceEvolution:
    def __init__(self, 
                 kokoro_path: str = "kokoro-v1.0.onnx",
                 voices_path: str = "voices-v1.0.bin",
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5,
                 output_dir: str = "output"):
        self.kokoro = Kokoro(kokoro_path, voices_path)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.output_dir = output_dir
        self.target_features = None
        self.population: List[VoiceGenome] = []
        self.fitness_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_target_audio(self, target_path: str):
        """Load and analyze target audio file"""
        audio, sr = librosa.load(target_path)
        self.target_features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)
        }
        
    def initialize_population(self, seed_voice_id: str):
        """Initialize population using a seed voice as starting point"""
        base_embedding = self.kokoro.get_voice_style(seed_voice_id)
        
        self.population = []
        for _ in range(self.population_size):
            # Create mutated versions of seed voice
            mutated = self._mutate_embedding(base_embedding.copy())
            self.population.append(VoiceGenome(embedding=mutated))
    
    def _mutate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Apply random mutations to voice embedding"""
        mutation_mask = np.random.random(embedding.shape) < self.mutation_rate
        mutations = np.random.normal(0, 0.1, embedding.shape)
        embedding[mutation_mask] += mutations[mutation_mask]
        return embedding
    
    def _calculate_fitness(self, audio: np.ndarray, sr: int) -> float:
        """Calculate fitness score using multiple audio features"""
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)
        }
        
        # Calculate duration penalty
        target_duration = self.target_features['mfcc'].shape[1] / sr  # Duration in seconds
        current_duration = len(audio) / sr
        duration_difference = abs(target_duration - current_duration)
        duration_penalty = np.exp(-duration_difference)  # Exponential penalty that decreases as difference increases
        
        # Calculate weighted similarities for each feature
        weights = {'mfcc': 0.4, 'spectral_centroid': 0.2, 
                  'spectral_rolloff': 0.2, 'zero_crossing_rate': 0.2}
        
        similarities = []
        for feature_name, weight in weights.items():
            target = self.target_features[feature_name]
            current = features[feature_name]
            
            # Normalize features to same length if needed
            if target.shape[1] != current.shape[1]:
                target = librosa.util.fix_length(target, size=min(target.shape[1], current.shape[1]), axis=1)
                current = librosa.util.fix_length(current, size=min(target.shape[1], current.shape[1]), axis=1)
            
            # Calculate similarity for this feature
            feature_similarities = [1 - cosine(t_frame, g_frame) 
                                  for t_frame, g_frame in zip(target.T, current.T)]
            similarities.append(np.mean(feature_similarities) * weight)
        
        # Apply duration penalty to final fitness score
        feature_fitness = sum(similarities)
        return feature_fitness * duration_penalty
    
    def _crossover(self, parent1: VoiceGenome, parent2: VoiceGenome) -> np.ndarray:
        """Create child embedding by combining two parents"""
        # Uniform crossover
        mask = np.random.random(parent1.embedding.shape) < 0.5
        child = np.where(mask, parent1.embedding, parent2.embedding)
        return child
    
    def evolve_generation(self, text: str):
        """Evolve one generation of voices"""
        # Evaluate current population
        for genome in self.population:
            audio, sr = self.kokoro.create(text, voice=genome.embedding)
            genome.fitness = self._calculate_fitness(audio, sr)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best fitness
        self.fitness_history.append(self.population[0].fitness)
        
        # Keep elite performers
        new_population = self.population[:self.elite_size]
        
        # Create rest of new population through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = random.choice(self.population[:10])  # Top 10 performers
            parent2 = random.choice(self.population[:10])
            
            # Create child through crossover
            child_embedding = self._crossover(parent1, parent2)
            
            # Mutate child
            child_embedding = self._mutate_embedding(child_embedding)
            
            new_population.append(VoiceGenome(embedding=child_embedding))
        
        self.population = new_population
    
    def get_best_voice(self) -> Tuple[np.ndarray, float]:
        """Return the best performing voice and its fitness score"""
        best = max(self.population, key=lambda x: x.fitness)
        return best.embedding, best.fitness

    def save_voice(self, voice_embedding: np.ndarray, generation: int) -> None:
        """Save the voice embedding to a .bin file"""
        output_file = os.path.join(self.output_dir, f"evolved_voice_gen_{generation}.bin")
        with open(output_file, "wb") as f:
            np.savez(f, evolved_voice=voice_embedding)
        print(f"Saved voice embedding to: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Voice Evolution Parameters')
    parser.add_argument('--population-size', type=int, default=50,
                      help='Size of the population (default: 50)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Mutation rate (default: 0.1)')
    parser.add_argument('--elite-size', type=int, default=5,
                      help='Number of elite members to preserve (default: 5)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for generated files (default: output)')
    parser.add_argument('--generations', type=int, default=50,
                      help='Number of generations to evolve (default: 50)')
    parser.add_argument('--target-audio', type=str, required=True,
                      help='Path to target audio file')
    parser.add_argument('--text', type=str, required=True,
                      help='Text to use for voice synthesis')
    args = parser.parse_args()

    # Initialize evolution with command line parameters
    evolution = VoiceEvolution(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        output_dir=args.output_dir
    )
    
    # Load target audio file
    evolution.load_target_audio(args.target_audio)
    
    # Initialize population using a seed voice
    evolution.initialize_population("af_heart")
    
    # Run evolution for N generations
    for generation in range(args.generations + 1):  # Changed to include the last generation
        evolution.evolve_generation(args.text)
        best_voice, best_fitness = evolution.get_best_voice()
        
        print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Generate and save best voice periodically (and on final generation)
        if generation % 5 == 0 or generation == args.generations:
            # Save audio file
            audio, sr = evolution.kokoro.create(args.text, voice=best_voice)
            output_wav = os.path.join(args.output_dir, f"evolution_gen_{generation}.wav")
            sf.write(output_wav, audio, sr)
            
            # Save voice embedding
            evolution.save_voice(best_voice, generation)
            
            # Force garbage collection to free memory
            gc.collect()
    
    # Clean up and free memory
    del evolution
    gc.collect()

if __name__ == "__main__":
    main()