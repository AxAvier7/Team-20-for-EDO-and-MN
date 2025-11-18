import numpy as np
import matplotlib.pyplot as plt

# Rango de parámetros mu
mu = np.linspace(-1.0, 2.0, 800)

# Inicializar arrays para las ramas de equilibrio (NaN donde no existen)
z_plus = np.full_like(mu, np.nan)
z_minus = np.full_like(mu, np.nan)

# Para mu >= 0, equilibria z = ±sqrt(mu)
mask = mu >= 0
z_plus[mask] = np.sqrt(mu[mask])
z_minus[mask] = -np.sqrt(mu[mask])

# Crear la figura
plt.figure(figsize=(8, 5))

# Dibujar ramas: z_plus estable (línea sólida), z_minus inestable (línea discontinua)
plt.plot(mu, z_plus, linestyle='-', linewidth=2, label='Equilibrio estable: $z=+\\sqrt{\\mu}$')
plt.plot(mu, z_minus, linestyle='--', linewidth=2, label='Equilibrio inestable: $z=-\\sqrt{\\mu}$')

# Añadir línea vertical en mu=0 para enfatizar el punto de bifurcación
plt.axvline(0, linestyle=':', linewidth=1)

# Anotaciones y etiquetas
plt.title('Diagrama de bifurcación para $\\dot z = \\mu - z^2$')
plt.xlabel('$\\mu$ (parámetro de control)')
plt.ylabel('$z$ (corriente)')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend(loc='upper left')

# Marcar el origen de la bifurcación
plt.scatter([0], [0], s=30)

# Guardar la figura
plt.savefig("diagrama_bifurcacion_mu_z.png", bbox_inches='tight', dpi=150)
plt.show()