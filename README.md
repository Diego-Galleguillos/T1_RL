# T1_RL

# Pregunta a)

Para poder correr el codigo de la pregunta a), se debe setear agent = SimpleAgent(num_of_arms, epsilon=epsilon)  en Main.py, ademas se debe setear epsilon_values = [0, 0.01, 0.1] tambien en Main.py. Usar bandit = BanditEnv(seed=run_id).

# Pregunta b) - no se corre nada nuevo

# Pregunta c) 
Se setea epsilon_values = [0, 0.1] en Main.py ya que ese experimetno no usa epsilon =0.01. Ademas se debe escoger agent = SimpleTrackingAgent(num_of_arms, epsilon=epsilon). Usar bandit = BanditEnv(seed=run_id). 

# Pregunta d) - no se corre nada nuevo

# Pregunta e) - no se corre nada nuevo

# Pregunta f)

Se setea epsilon_values = [0,1] ( no usamos epsilon pero reurilizamos el array para que el codigo sea mas facil de correr para quien lo corrija, 0 es sin baseline con Rt = 0 y 1 es con baseline donde Rt se calcula en cada iteracion) en Main.py . Ademas se debe escoger agent = GradientAgent(num_of_arms, baseline=epsilon). Finalmente tambien es necesario cambiar la media de los bandits cambiando bandit = BanditEnv(seed=run_id, mean=4).

# Pregunta g) - no se corre nada nuevo