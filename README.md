# Taller Tercer Corte IA
* Yefri Stiven Barrero Solano - 2320392

## Parte 1: Aprendizaje por Refuerzo (Reinforcement Learning - RL)

### a) ¿Cómo puede un agente aprender a tomar decisiones óptimas en un entorno incierto?

El aprendizaje por refuerzo es un paradigma de machine learning donde un **agente** interactúa con un **entorno** desconocido tratando de maximizar una recompensa acumulada a largo plazo.

Elementos clave del proceso:
- El agente observa el **estado** actual del entorno (state).
- Elige una **acción** según una **política** π (policy).
- El entorno responde con un nuevo estado y una **recompensa** inmediata (reward).
- El agente actualiza su conocimiento para mejorar futuras decisiones.

El aprendizaje ocurre mediante el principio de **prueba y error**: el agente explora acciones nuevas (exploration) y también aprovecha lo que ya sabe que funciona (exploitation). 

La incertidumbre del entorno se maneja mediante:
- Estimación de la **función de valor** (value function): cuánto vale estar en un estado o tomar una acción.
- Uso de **descuento** (γ ∈ [0,1)) para priorizar recompensas inmediatas sobre las muy lejanas.
- Técnicas como Q-Learning, SARSA, Policy Gradient, Actor-Critic, etc., que convergen (bajo ciertas condiciones) a una **política óptima** π* que maximiza la recompensa esperada.

### b) Tipos de algoritmos de aprendizaje por refuerzo y sus arquitecturas

#### Clasificación principal

| Categoría                  | Basado en valor (Value-based) | Basado en política (Policy-based) | Actor-Critic (híbrido) |
|----------------------------|-------------------------------|-----------------------------------|-------------------------|
| Ejemplos                   | Q-Learning, SARSA, Deep Q-Network (DQN) | REINFORCE, PPO, TRPO            | A2C, A3C, DDPG, TD3, SAC |
| Qué aprende directamente   | Valor de acciones (Q(s,a))   | Política π(a|s) directamente    | Ambos: Actor (política) + Critic (valor) |
| Ventaja                    | Estable en entornos discretos | Maneja espacios de acción continuos | Buena estabilidad y rendimiento |
| Desventaja                 | Problemas con acciones continuas | Alta varianza en gradientes    | Más complejo de implementar |

#### Elementos fundamentales de cualquier algoritmo RL (MDP - Markov Decision Process)

1. **S**: Conjunto de estados
2. **A**: Conjunto de acciones
3. **P(s'|s,a)**: Probabilidad de transición (dinámica del entorno, usualmente desconocida)
4. **R(s,a,s')** o **R(s,a)**: Recompensa
5. **γ**: Factor de descuento (0 ≤ γ < 1)
6. **Política π(a|s)**: Estrategia del agente
7. **Función de valor Vπ(s)**: Recompensa esperada futura desde s siguiendo π
8. **Función acción-valor Qπ(s,a)**: Recompensa esperada tomando a en s y luego siguiendo π

#### Arquitecturas más importantes

| Algoritmo       | Tipo               | Descripción clave                                                                                       | Componentes principales                                     |
|-----------------|--------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Q-Learning      | Value-based (off-policy) | Actualiza tabla Q(s,a) → Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]                                   | Tabla Q, ε-greedy para exploración                          |
| SARSA           | Value-based (on-policy)  | Igual que Q-Learning pero usa la acción realmente tomada: Q(s,a) ← r + γ Q(s',a')                     | Tabla Q, política ε-greedy                                  |
| DQN             | Deep Value-based   | Usa red neuronal profunda para aproximar Q(s,a). Experience Replay + Target Network                    | CNN/MLP, buffer de experiencia, red objetivo fija           |
| REINFORCE       | Policy Gradient    | Gradiente de Monte-Carlo: ∇θ log π(a|s;θ) · G_t  (G_t = retorno)                                       | Red que parametriza π(a|s;θ), cálculo de retornos           |
| Actor-Critic    | Híbrido            | Actor = política, Critic = estima V(s) o Q(s,a) para reducir varianza                                   | Red Actor, red Critic, ventaja A(s,a)                       |
| A2C / A3C       | Actor-Critic asíncrono | Múltiples workers exploran en paralelo (A3C) o sincronizados (A2C)                                     | Múltiples entornos, ventaja generalizada                   |
| PPO             | Policy Gradient (on-policy) | clipped surrogate objective → evita pasos de política muy grandes                                     | Ratio de probabilidades, clipping, múltiples épocas por batch |
| DDPG            | Actor-Critic (off-policy, continuo) | Extensión de DQN para acciones continuas + deterministic policy gradient                             | Actor determinístico μ(s), Critic Q(s,a), ruido Ornstein-Uhlenbeck |
| TD3             | Mejora de DDPG     | Doble Critic, delayed policy updates, target policy smoothing                                          | Dos Q-networks, actualización retardada del actor           |
| SAC             | Actor-Critic + Entropía | Maximiza recompensa + entropía de la política (exploración automática)                                 | Regularización por entropía, dos Q-networks (soft min)      |

### c) Usos industriales del aprendizaje por refuerzo

| Industria               | Aplicación real                                                                 | Algoritmo típico usado        |
|-------------------------|---------------------------------------------------------------------------------|-------------------------------|
| Robótica                | Control de brazos robóticos, drones, locomoción de robots cuadrúpedos           | PPO, SAC, DDPG                |
| Videojuegos & Simulación| Entrenamiento de agentes en juegos complejos (Dota 2 - OpenAI Five, StarCraft - AlphaStar) | PPO + self-play               |
| Finanzas                | Trading algorítmico, gestión de portafolios, optimización de órdenes            | Deep RL (DDPG, PPO)           |
| Recomendación           | Sistemas de recomendación multi-arm bandit contextual y secuencial             | Contextual Bandits, PPO       |
| Energía                 | Control de redes eléctricas inteligentes, optimización de baterías             | Multi-agent RL, SAC           |
| Manufactura             | Optimización de líneas de producción, scheduling                                | MARL (Multi-Agent RL)         |
| Salud                   | Dosificación dinámica de medicamentos, optimización de tratamientos oncológicos | PPO, SAC                      |
| Publicidad digital      | Optimización de pujas en tiempo real (RTB)                                      | Contextual Bandits, DQN       |
| Vehículos autónomos     | Planificación de trayectorias, control de bajo nivel                            | Model-based RL, SAC           |

## Parte 2: Clasificador Bayesiano Naïve de Spam con Teorema de Bayes

### Datos conocidos
- P(Spam) = 0.30 → Prior
- P(No Spam) = 0.70
- P("gratis" | Spam) = 0.80
- P("gratis" | No Spam) = 0.10

Queremos: **P(Spam | "gratis")**

### Teorema de Bayes

```
P(Spam | "gratis") = [P("gratis" | Spam) × P(Spam)] / P("gratis")
```

Primero calculamos P("gratis") usando la regla de probabilidad total:

```
P("gratis") = P("gratis" | Spam) × P(Spam) + P("gratis" | No Spam) × P(No Spam)
            = (0.80 × 0.30) + (0.10 × 0.70)
            = 0.24 + 0.07
            = 0.31
```

Entonces:

```
P(Spam | "gratis") = (0.80 × 0.30) / 0.31 = 0.24 / 0.31 ≈ 0.7742 → 77.42%
```

### Algoritmo en Python

```python
def es_spam_con_gratis():
    # Probabilidades a priori y condicionales
    p_spam = 0.30
    p_no_spam = 0.70
    p_gratis_dado_spam = 0.80
    p_gratis_dado_no_spam = 0.10
    
    # Evidencia
    p_gratis = (p_gratis_dado_spam * p_spam) + (p_gratis_dado_no_spam * p_no_spam)
    
    # Posterior
    p_spam_dado_gratis = (p_gratis_dado_spam * p_spam) / p_gratis
    
    print(f"Probabilidad de que sea spam dado que contiene 'gratis': {p_spam_dado_gratis:.4f} ({p_spam_dado_gratis*100:.2f}%)")
    
    # Umbral típico: >50% → clasificar como spam
    if p_spam_dado_gratis > 0.5:
        return True, p_spam_dado_gratis
    else:
        return False, p_spam_dado_gratis

resultado, probabilidad = es_spam_con_gratis()
print("Clasificación:" , "SPAM" if resultado else "NO SPAM")
```

Resultado: **77.42% → se clasifica como SPAM**

En la práctica se usan muchas más palabras y se asume independencia condicional (Naïve Bayes), lo que permite multiplicar probabilidades.

## Parte 3: Algoritmos de IA más utilizados en academia e industria

| Categoría                | Algoritmo / Arquitectura               | Año aprox. popularización | Uso principal industria                     | Uso principal academia                     | Características clave                                    |
|--------------------------|----------------------------------------|---------------------------|---------------------------------------------|--------------------------------------------|----------------------------------------------------------|
| Redes Neuronales         | Transformers                           | 2017                      | LLMs (ChatGPT, Claude, Grok, Llama)         | Investigación en NLP, visión, multimodal   | Atención, escalabilidad, preentrenamiento masivo         |
| Grandes Modelos Lenguaje | GPT-4o, Claude 3.5, Llama-3/4, Grok-4, Gemini 1.5 | 2022-2025               | Chatbots, copilots, generación de código    | Reasoning, agentes, alineación             | Billones de parámetros, RLHF, MoE (Mixture of Experts) |
| Visión por Computadora   | Vision Transformers (ViT), ConvNeXt   | 2020-2023                 | Reconocimiento facial, vehículos autónomos  | Análisis médico, robótica                  | Atención global, mejor escalabilidad que CNNs            |
| Difusión                | Stable Diffusion 3, Flux, SDXL, DALL-E 4 | 2022-2025               | Generación de imágenes y vídeo comerciales  | Arte generativo, edición de imágenes       | Modelo de difusión en espacio latente, alta calidad      |
| RL + LLMs                | RLHF, PPO en LLMs                      | 2022-2025                 | Alineación de modelos (ChatGPT)             | Agentes autónomos (Auto-GPT, Voyager)      | Refuerzo con feedback humano                             |
| Grafos                   | Graph Neural Networks (GNN)            | 2018-2025                 | Recomendación (Pinterest, Amazon), fármacos | Descubrimiento de fármacos, redes sociales | Manejo nativo de datos relacionales                      |
| Eficiencia               | Mixture of Experts (MoE)               | 2021-2025                 | Modelos gigantes eficientes (Grok-1, Mixtral, DeepSeek) | Escalado eficiente                        | Activación sparse, mismo rendimiento con menos FLOPs     |
| Multimodal               | CLIP, Flamingo, GPT-4V, Gemini         | 2022-2025                 | Asistentes con visión + lenguaje            | Robótica, búsqueda visual                  | Entrenamiento contrastivo imagen-texto                   |
| Agentes                  | Auto-GPT, BabyAGI, LangChain + ReAct   | 2023-2025                 | Automatización de tareas complejas          | Investigación en planificación y razonamiento | Bucle de pensamiento-acción-observación               |
| Modelos abiertos         | Llama-3.1 405B, Qwen2, Mistral Large   | 2024-2025                 | Empresas que no pueden pagar APIs cerradas | Fine-tuning, investigación responsable     | Alto rendimiento, descargables, permisos comerciales     |
