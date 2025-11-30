import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Ecuation:
    def __init__(self, ecuacion_diferencial):
        self.ecuacion = ecuacion_diferencial
    
    def graph_isoclines(self, xRange=(-5, 5), yRange=(-5, 5), Slopes=None, Density=20, FigureSize=(12, 10)):
        
        # Malla de puntos
        x = np.linspace(xRange[0], xRange[1], Density)
        y = np.linspace(yRange[0], yRange[1], Density)
        X, Y = np.meshgrid(x, y)
        
        # Calcular las Slopes en cada punto
        U = np.ones_like(X)  # Componente x del vector (dy/dx)
        V = self.ecuacion(X, Y)  # Componente y del vector (dy/dx)
        
        # Normalizar los vectores
        norm = np.sqrt(U**2 + V**2)
        U = U / norm
        V = V / norm
        
        # Crear la figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FigureSize)
        
        # Gráfico 1: Campo de pendientes
        ax1.quiver(X, Y, U, V, color='blue', alpha=0.7, scale=20, width=0.005)
        ax1.set_title('Slopes Field', fontsize=14)
        ax1.set_xlabel('t', fontsize=12)
        ax1.set_ylabel('Q', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Gráfico 2: Isoclinas
        if Slopes is None:
            # Calcular las pendientes con el rango de valores
            slopes_values = np.linspace(np.min(V), np.max(V), 8)
            Slopes = slopes_values[::2]  # Tomar algunos valores representativos
        
        # Graficar las isoclinas
        for m in Slopes:
            # Para cada pendiente m, resolver f(x,y) = m
            Z = self.ecuacion(X, Y) - m
            contorno = ax2.contour(X, Y, Z, levels=[0], colors=plt.cm.viridis(m/max(Slopes)))
            ax2.clabel(contorno, inline=True, fontsize=8, fmt=f'm={m:.2f}')
        
        ax2.set_title('Isoclines', fontsize=14)
        ax2.set_xlabel('t', fontsize=12)
        ax2.set_ylabel('Q', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def add_sol(self, Initial_Conditions, xRange=(-5, 5), t_Points=100):
        
        x_sol = np.linspace(xRange[0], xRange[1], t_Points)
        
        plt.figure(figsize=(10, 8))
        
        # Graficar campo de pendientes
        self._add_slopes_field(xRange)
        
        # Graficar soluciones
        colours = plt.cm.Set1(np.linspace(0, 1, len(Initial_Conditions)))
        
        for i, y0 in enumerate(Initial_Conditions):
            # Resolver la ecuación diferencial
            def system_ecuation(y, x):
                return self.ecuacion(x, y)
            
            solucion = odeint(system_ecuation, y0, x_sol)
            
            plt.plot(x_sol, solucion[:, 0], 
                    color=colours[i], 
                    linewidth=2, 
                    label=f'y({xRange[0]}) = {y0}')
        
        
        plt.title('EDOs Solution', fontsize=14)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Q', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.show()
        
    # GRÁFICAS DE LOS MÉTODOS
    def graph_euler(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        #Calcula y grafica la solución usando el método de Euler
        x_vals, y_vals = self.euler_method(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._add_slopes_field(x_range)
        plt.plot(x_vals, y_vals, 'o-', linewidth=2, label=f'Euler (h={h})', markersize=6, color='red')
        self._graph_config('Euler Method')
        plt.show()
                
        return x_vals, y_vals

    def graph_improved_euler(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        #Calcula y grafica la solución usando el método de Euler Mejorado
        x_vals, y_vals = self.improved_euler(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._add_slopes_field(x_range)
        plt.plot(x_vals, y_vals, 's-', linewidth=2, label=f'Euler Mejorado (h={h})', markersize=5, color='green')
        self._graph_config('Improved Euler Method')
        plt.show()
        
        return x_vals, y_vals

    def graph_runge_kutta(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        #Calcula y grafica la solución usando el método de Runge-Kutta 4
        x_vals, y_vals = self.runge_kutta_4(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._add_slopes_field(x_range)
        plt.plot(x_vals, y_vals, '^-', linewidth=2, label=f'Runge-Kutta 4 (h={h})', markersize=5, color='blue')
        self._graph_config('Runge-Kutta 4 Method')
        plt.show()
        
        return x_vals, y_vals

    # CÁLCULO DE MÉTODOS
    def euler_method(self, y0, x_range, h=0.1):
        x_vals = np.arange(x_range[0], x_range[1] + h, h)
        y_vals = np.zeros(len(x_vals))
        y_vals[0] = y0
        
        for i in range(1, len(x_vals)):
            d = self.ecuacion(x_vals[i-1], y_vals[i-1])
            
            y_vals[i] = y_vals[i-1] + h * d
        
        return x_vals, y_vals

    def improved_euler(self, y0, x_range, h=0.1):
        x_vals = np.arange(x_range[0], x_range[1] + h, h)
        y_vals = np.zeros(len(x_vals))
        y_vals[0] = y0
        
        for i in range(1, len(x_vals)):
            y_Euler = y_vals[i-1] + h * self.ecuacion(x_vals[i-1], y_vals[i-1])
            y_vals[i] = y_vals[i-1] + h * 0.5 * (
                self.ecuacion(x_vals[i-1], y_vals[i-1]) + 
                self.ecuacion(x_vals[i], y_Euler)
            )
        
        return x_vals, y_vals

    def runge_kutta_4(self, y0, x_range, h=0.1):
        n_points = int((x_range[1] - x_range[0]) / h) + 1
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = np.zeros(n_points)
        y_vals[0] = y0
        
        for i in range(1, len(x_vals)):
            h_actual = x_vals[i] - x_vals[i-1]
            x_actual = x_vals[i-1]
            y_actual = y_vals[i-1]
            
            k1 = h_actual * self.ecuacion(x_actual, y_actual)
            k2 = h_actual * self.ecuacion(x_actual + h_actual/2, y_actual + k1/2)
            k3 = h_actual * self.ecuacion(x_actual + h_actual/2, y_actual + k2/2)  
            k4 = h_actual * self.ecuacion(x_actual + h_actual, y_actual + k3)
            
            y_vals[i] = y_actual + (k1 + 2*k2 + 2*k3 + k4) / 6
    
        return x_vals, y_vals

    # FUNCIONES AUXILIARES PRIVADAS
    def _add_slopes_field(self, x_range, y_range=None, density=15):
        #Agrega el campo de pendientes como fondo
        if y_range is None:
            y_range = (-5, 5)
        
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        V = self.ecuacion(X, Y)
        
        U = np.ones_like(X)
        norm = np.sqrt(U**2 + V**2)
        U = U / norm
        V = V / norm
        
        plt.quiver(X, Y, U, V, color='gray', alpha=0.5, scale=20, width=0.004)

    def _graph_config(self, titulo):
        #Configura los elementos comunes de las gráficas
        plt.title(titulo, fontsize=14)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Q', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
def calcular_orden_convergencia(metodo_func, y0, x_range, h_values, solucion_analitica):
    """
    Calcula el orden de convergencia de un método numérico
    """
    errores = []
    
    for h in h_values:
        # Calcular solución numérica
        x_vals, y_vals = metodo_func(y0, x_range, h)
        
        # Calcular solución analítica en los mismos puntos
        y_analitica = solucion_analitica(x_vals, 0)
        
        # Calcular error máximo
        error = np.max(np.abs(y_vals - y_analitica))
        errores.append(error)
    
    # Calcular órdenes de convergencia
    ordenes = []
    for i in range(1, len(errores)):
        if errores[i] > 0 and errores[i-1] > 0:
            orden = np.log(errores[i-1] / errores[i]) / np.log(h_values[i-1] / h_values[i])
            ordenes.append(orden)
    
    return h_values, errores, ordenes

      
    # Parámetros del circuito RC
Eo = 110
C = 0.05
R = 50

p = 9.9
    
def solucion_analitica(t, Q):
    return Eo * C * (1 - np.exp(-t / (R * C)))
    
def EDO(t, Q):
    return (Eo - Q/C) / R  
     
numerica = Ecuation(EDO)         
    
    # Graficar solución analítica
print("Solución Analítica:")
numerica.graph_isoclines(xRange=(0, 10), yRange=(-1, 6), Slopes=[0, 0.5, 1, 1.5, 2])
numerica.add_sol([0], xRange=(0, 10))
    
    # Comparar con métodos numéricos
x1, y1 = numerica.euler_method(y0=0, x_range=(0, p), h=0.1)
    
x2, y2 = numerica.improved_euler(y0=0, x_range=(0, p), h=0.1)
    
x3, y3 = numerica.runge_kutta_4(y0=0, x_range=(0, p), h=0.1)
    
    # COMPARACIÓN DIRECTA
plt.figure(figsize=(12, 8))
    
    # Solución analítica
t_analitico = np.linspace(0, 10, 100)
Q_analitico = solucion_analitica(t_analitico, 0)
plt.plot(t_analitico, Q_analitico, 'k-', linewidth=3, label='Analitical Solution')
    
    # Métodos numéricos
plt.plot(x1, y1, 'ro-', markersize=4, label=f'Euler (h=0.5)')
plt.plot(x2, y2, 'gs-', markersize=4, label=f'Improved Euler (h=0.5)')
plt.plot(x3, y3, 'b^-', markersize=4, label=f'Runge-Kutta 4 (h=0.5)')
    
plt.title('Numerical Methods vs Analitical Solution Comparison', fontsize=14)
plt.xlabel('t', fontsize=12)
plt.ylabel('Q', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
    
    # Mostrar valores finales para comparación para 
print(f"\nValor final (t=10):")
print(f"Analítico: {solucion_analitica(p, 0):.6f}")
print(f"Euler: {y1[-1]:.6f}, {y1[-2]:.6f}, {y1[-3]:.6f}")
print(f"Euler Mejorado: {y2[-1]:.6f}, {y2[-2]:.6f}, {y2[-3]:.6f}")
print(f"Runge-Kutta 4: {y3[-1]:.6f}, {y3[-2]:.6f}, {y3[-3]:.6f}")\
    
    
 # Diferentes tamaños de paso para el análisis
h_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
x_range = (0, 10)
y0 = 0
    
    
# Calcular órdenes de convergencia para cada método
print("\n1. MÉTODO DE EULER:")
h_euler, errores_euler, ordenes_euler = calcular_orden_convergencia(
    numerica.euler_method, y0, x_range, h_values, solucion_analitica
    )
print(f"Errores: {[f'{e:.6f}' for e in errores_euler]}")
print(f"Órdenes de convergencia: {[f'{o:.4f}' for o in ordenes_euler]}")
print(f"Orden promedio: {np.mean(ordenes_euler):.4f}")
    
print("\n2. MÉTODO DE EULER MEJORADO:")
h_mejorado, errores_mejorado, ordenes_mejorado = calcular_orden_convergencia(
    numerica.improved_euler, y0, x_range, h_values, solucion_analitica
    )
print(f"Errores: {[f'{e:.6f}' for e in errores_mejorado]}")
print(f"Órdenes de convergencia: {[f'{o:.4f}' for o in ordenes_mejorado]}")
print(f"Orden promedio: {np.mean(ordenes_mejorado):.4f}")
    
print("\n3. MÉTODO DE RUNGE-KUTTA 4:")
h_rk4, errores_rk4, ordenes_rk4 = calcular_orden_convergencia(
    numerica.runge_kutta_4, y0, x_range, h_values, solucion_analitica
    )
print(f"Errores: {[f'{e:.6f}' for e in errores_rk4]}")
print(f"Órdenes de convergencia: {[f'{o:.4f}' for o in ordenes_rk4]}")
print(f"Orden promedio: {np.mean(ordenes_rk4):.4f}")