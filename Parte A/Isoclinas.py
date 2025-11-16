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
        U = np.ones_like(X)  # Componente x del vector (siempre 1 para dy/dx)
        V = self.ecuacion(X, Y)  # Componente y del vector (dy/dx)
        
        # Normalizar los vectores
        magnitud = np.sqrt(U**2 + V**2)
        U = U / magnitud
        V = V / magnitud
        
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
            # Calcular las pendientes automáticamente basado en el rango de valores
            valores_pendiente = np.linspace(np.min(V), np.max(V), 8)
            Slopes = valores_pendiente[::2]  # Tomar algunos valores representativos
        
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
    
    def agregar_soluciones(self, Initial_Conditions, xRange=(-5, 5), t_Points=100):
        
        x_sol = np.linspace(xRange[0], xRange[1], t_Points)
        
        plt.figure(figsize=(10, 8))
        
        # Graficar campo de pendientes
        x = np.linspace(xRange[0], xRange[1], 20)
        y = np.linspace(xRange[0], xRange[1], 20)
        X, Y = np.meshgrid(x, y)
        U = np.ones_like(X)
        V = self.ecuacion(X, Y)
        magnitud = np.sqrt(U**2 + V**2)
        U = U / magnitud
        V = V / magnitud
        
        plt.quiver(X, Y, U, V, color='gray', alpha=0.5, scale=20, width=0.005)
        
        # Graficar soluciones
        colores = plt.cm.Set1(np.linspace(0, 1, len(Initial_Conditions)))
        
        for i, y0 in enumerate(Initial_Conditions):
            # Resolver la ecuación diferencial
            def ecuacion_sistema(y, x):
                return self.ecuacion(x, y)
            
            solucion = odeint(ecuacion_sistema, y0, x_sol)
            
            plt.plot(x_sol, solucion[:, 0], 
                    color=colores[i], 
                    linewidth=2, 
                    label=f'y({xRange[0]}) = {y0}')
        
        plt.title('Soluciones de la Ecuación Diferencial', fontsize=14)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Q', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.show()
        
    # GRÁFICAS DE LOS MÉTODOS
    def graficar_euler(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        """Calcula y grafica la solución usando el método de Euler"""
        x_vals, y_vals = self.euler_method(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._agregar_campo_pendientes(x_range)
        plt.plot(x_vals, y_vals, 'o-', linewidth=2, label=f'Euler (h={h})', markersize=6, color='red')
        self._configurar_grafica('Solución - Método de Euler')
        plt.show()
        
        return x_vals, y_vals

    def graficar_euler_mejorado(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        """Calcula y grafica la solución usando el método de Euler Mejorado"""
        x_vals, y_vals = self.euler_mejorado(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._agregar_campo_pendientes(x_range)
        plt.plot(x_vals, y_vals, 's-', linewidth=2, label=f'Euler Mejorado (h={h})', markersize=5, color='green')
        self._configurar_grafica('Solución - Método de Euler Mejorado')
        plt.show()
        
        return x_vals, y_vals

    def graficar_runge_kutta(self, y0, x_range=(-5, 5), h=0.1, figsize=(10, 8)):
        """Calcula y grafica la solución usando el método de Runge-Kutta 4"""
        x_vals, y_vals = self.runge_kutta_4(y0, x_range, h)
        
        plt.figure(figsize=figsize)
        self._agregar_campo_pendientes(x_range)
        plt.plot(x_vals, y_vals, '^-', linewidth=2, label=f'Runge-Kutta 4 (h={h})', markersize=5, color='blue')
        self._configurar_grafica('Solución - Método de Runge-Kutta 4')
        plt.show()
        
        return x_vals, y_vals

    # CÁLCULO DE MÉTODOS
    def euler_method(self, y0, x_range, h=0.1):
        x_vals = np.arange(x_range[0], x_range[1] + h, h)
        y_vals = np.zeros(len(x_vals))
        y_vals[0] = y0
        
        for i in range(1, len(x_vals)):
            derivada = self.ecuacion(x_vals[i-1], y_vals[i-1])
            
            y_vals[i] = y_vals[i-1] + h * derivada
        
        return x_vals, y_vals

    def euler_mejorado(self, y0, x_range, h=0.1):
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
        x_vals = np.arange(x_range[0], x_range[1] + h, h)
        y_vals = np.zeros(len(x_vals))
        y_vals[0] = y0
        
        for i in range(1, len(x_vals)):
            k1 = self.ecuacion(x_vals[i-1], y_vals[i-1])
            k2 = self.ecuacion(x_vals[i-1] + h/2, y_vals[i-1] + k1/2)
            k3 = self.ecuacion(x_vals[i-1] + h/2, y_vals[i-1] + k2/2)
            k4 = self.ecuacion(x_vals[i-1] + h, y_vals[i-1] + k3)
            
            y_vals[i] = y_vals[i-1] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return x_vals, y_vals

    # FUNCIONES AUXILIARES PRIVADAS
    def _agregar_campo_pendientes(self, x_range, y_range=None, density=15):
        """Agrega el campo de pendientes como fondo"""
        if y_range is None:
            # Estimar un rango de y razonable basado en la ecuación
            y_range = (-5, 5)
        
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        V = self.ecuacion(X, Y)
        
        U = np.ones_like(X)
        magnitud = np.sqrt(U**2 + V**2)
        U = U / magnitud
        V = V / magnitud
        
        plt.quiver(X, Y, U, V, color='gray', alpha=0.5, scale=20, width=0.004)

    def _configurar_grafica(self, titulo):
        """Configura los elementos comunes de las gráficas"""
        plt.title(titulo, fontsize=14)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Q', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)


if __name__ == "__main__":        
    # Parámetros del circuito RC
    Eo = 110
    C = 0.05
    R = 50
    
    def solucion_analitica(t, Q):
        return Eo * C * (1 - np.exp(-t / (R * C)))
    
    def EDO(t, Q):
        return (Eo - Q/C) / R  
     
    numerica = Ecuation(EDO)         
    
    # Graficar solución analítica
    print("Solución Analítica:")
    numerica.graph_isoclines(xRange=(0, 10), yRange=(-1, 6), Slopes=[0, 0.5, 1, 1.5, 2])
    numerica.agregar_soluciones([0], xRange=(0, 10))
    
    # Comparar con métodos numéricos
    print("\nMétodo de Euler:")
    x1, y1 = numerica.graficar_euler(y0=0, x_range=(0, 10), h=0.1)
    
    print("\nMétodo de Euler Mejorado:")
    x2, y2 = numerica.graficar_euler_mejorado(y0=0, x_range=(0, 10), h=0.1)
    
    print("\nMétodo de Runge-Kutta 4:")
    x3, y3 = numerica.graficar_runge_kutta(y0=0, x_range=(0, 10), h=0.3)
    
    # COMPARACIÓN DIRECTA
    plt.figure(figsize=(12, 8))
    
    # Solución analítica
    t_analitico = np.linspace(0, 10, 100)
    Q_analitico = solucion_analitica(t_analitico, 0)
    plt.plot(t_analitico, Q_analitico, 'k-', linewidth=3, label='Solución Analítica')
    
    # Métodos numéricos
    plt.plot(x1, y1, 'ro-', markersize=4, label=f'Euler (h=0.5)')
    plt.plot(x2, y2, 'gs-', markersize=4, label=f'Euler Mejorado (h=0.5)')
    plt.plot(x3, y3, 'b^-', markersize=4, label=f'Runge-Kutta 4 (h=0.5)')
    
    plt.title('Comparación: Solución Analítica vs Métodos Numéricos', fontsize=14)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Q', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Mostrar valores finales para comparación
    print(f"\nValor final (t=10):")
    print(f"Analítico: {solucion_analitica(10, 0):.6f}")
    print(f"Euler: {y1[-1]:.6f}")
    print(f"Euler Mejorado: {y2[-1]:.6f}")
    print(f"Runge-Kutta 4: {y3[-1]:.6f}")