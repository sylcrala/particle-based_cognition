from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
import time
import math

from apis.api_registry import api

class FieldVisualizerWidget(Static):
    def __init__(self):
        super().__init__("Particle Field - Status: Offline")
        self.field_api = api.get_api("_agent_field")
        # proposed visualization:
        # x, y, z - Spatial
        # w - Temporal
        # f, v - Emotional
        # q - Certainty
        self.logger = api.get_api("logger")


        self.render_cache = {}
        self.connection_threshold = 0.3
        self.last_render_time = 0
        self.field_width = 80
        self.field_height = 20

    def on_mount(self):
        self.set_interval(1.0, self.update_display)

    def retry_api(self):
        self.field_api = api.get_api("_agent_field")
        self.logger.log("Particle Field API re-attempted", level="SYSTEM")

    async def update_display(self):
        if not self.field_api:
            self.update("❌ Particle Field API not available")
            self.retry_api()
            return
            
        try:
            current_particles = await self.field_api.get_all_particles()
            
            # Update cache for changed particles
            for particle in current_particles:
                if particle.last_updated > self.last_render_time:
                    self.render_cache[particle.id] = particle.render_particle()

            # Build ASCII visualization
            visualization = self.build_visualization_from_cache()
            self.update(visualization)
            self.last_render_time = time.time()
            
        except Exception as e:
            self.update(f"❌ Visualization error: {e}")
    
    def build_visualization_from_cache(self):
        """Build ASCII field visualization from cached render data"""
        if not self.render_cache:
            return "🌌 Particle Field - Empty\n\n   No active particles detected"
        
        # Create 2D field array for ASCII rendering
        field = [[' ' for _ in range(self.field_width)] for _ in range(self.field_height)]
        
        # Track particles for connection rendering
        particle_positions = {}
        
        # Render particles first
        for particle_id, render_data in self.render_cache.items():
            x, y, z = render_data['position']
            
            # Project 3D to 2D screen coordinates
            screen_x = int((x * 0.8 + 0.1) * self.field_width)  # Scale and center
            screen_y = int((y * 0.8 + 0.1) * self.field_height)
            
            # Clamp to field boundaries
            screen_x = max(0, min(self.field_width - 1, screen_x))
            screen_y = max(0, min(self.field_height - 1, screen_y))
            
            # Store position for connection rendering
            particle_positions[particle_id] = (screen_x, screen_y)
            
            # Choose particle character based on quantum state and size
            particle_char = self._get_particle_character(render_data)
            
            # Place particle in field
            field[screen_y][screen_x] = particle_char
        
        # Render entanglement connections (thin lines)
        self._render_connections(field, particle_positions)
        
        # Convert field to string with header
        return self._field_to_string(field)
    
    def _get_particle_character(self, render_data):
        """Choose ASCII character based on particle type and 7D render data"""
        quantum_state = render_data['quantum_state']
        size = render_data['size']
        particle_type = render_data.get('type', 'unknown')
        current_time = time.time()
        
        # Define character sets for each particle type
        if particle_type == 'lingual':
            # Lingual particles: Language/communication focused
            # Use speech-bubble and text-like characters
            if size < 0.5:
                base_chars = ['◌', '◦', '○']  # Small lingual particles
            elif size < 1.0:
                base_chars = ['○', '◎', '⊙']  # Medium lingual particles
            else:
                base_chars = ['⊙', '⊚', '⬡']  # Large/mature lingual particles
        
        elif particle_type == 'memory':
            # Memory particles: Storage/retrieval focused  
            # Use solid, geometric characters representing stored information
            if size < 0.5:
                base_chars = ['·', '▪', '■']  # Small memory particles
            elif size < 1.0:
                base_chars = ['■', '▬', '▲']  # Medium memory particles
            else:
                base_chars = ['▲', '♦', '⬢']  # Large/important memory particles
        
        else:
            # Fallback for unknown types
            if size < 0.5:
                base_chars = ['·', '◦', '○']
            elif size < 1.0:
                base_chars = ['○', '◉', '●']
            else:
                base_chars = ['●', '◉', '⬢']
        
        # Quantum state affects character choice
        certainty = quantum_state['opacity']
        
        if quantum_state['collapse_indicator']:
            # Recently collapsed - use strongest character for type
            if particle_type == 'lingual':
                return '⬡' if size > 1.0 else '⊙'
            elif particle_type == 'memory':
                return '⬢' if size > 1.0 else '▲'
            else:
                return '⬢' if size > 1.0 else '●'
        
        elif quantum_state['animation'] == 'shimmer':
            # Shimmering superposition - alternate between normal and faded
            shimmer_cycle = int(current_time * 3) % 2
            if shimmer_cycle:
                char_index = min(len(base_chars)-1, int(certainty * len(base_chars)))
                return base_chars[char_index]
            else:
                # Faded shimmer state - use type-specific fade character
                if particle_type == 'lingual':
                    return '◌'  # Hollow circle for lingual fade
                elif particle_type == 'memory':
                    return '▫'  # Hollow square for memory fade
                else:
                    return '◦'  # Generic fade
        
        elif quantum_state['ghost_trails']:
            # Highly uncertain - use ghost characters specific to type
            if particle_type == 'lingual':
                return '◌' if certainty < 0.3 else '◦'
            elif particle_type == 'memory':
                return '▫' if certainty < 0.3 else '▪'
            else:
                return '◦' if certainty < 0.3 else '○'
        
        else:
            # Normal state - use size and certainty based character
            char_index = min(len(base_chars)-1, int(certainty * len(base_chars)))
            return base_chars[char_index]
    
    def _render_connections(self, field, particle_positions):
        """Render connection lines between entangled particles with type-aware styling"""
        positions_by_type = {}
        particle_types = {}
        
        # Group particles by type for intelligent connection rendering
        for particle_id, render_data in self.render_cache.items():
            if particle_id in particle_positions:
                particle_type = render_data.get('type', 'unknown')
                positions_by_type.setdefault(particle_type, []).append(particle_positions[particle_id])
                particle_types[particle_positions[particle_id]] = particle_type
        
        positions = list(particle_positions.values())
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                x1, y1 = pos1
                x2, y2 = pos2
                
                # Get particle types for connection styling
                type1 = particle_types.get(pos1, 'unknown')
                type2 = particle_types.get(pos2, 'unknown')
                
                # Only connect nearby particles (to avoid visual clutter)
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if distance < 12 and distance > 1:  # Close but not touching
                    # Choose connection style based on particle types
                    connection_style = self._get_connection_style(type1, type2, distance)
                    self._draw_connection_line(field, x1, y1, x2, y2, connection_style)
    
    def _get_connection_style(self, type1, type2, distance):
        """Determine connection line style based on particle types"""
        if type1 == 'memory' and type2 == 'memory':
            # Memory-to-memory: solid lines (information links)
            return {'horizontal': '═', 'vertical': '║', 'strength': 'strong'}
        elif type1 == 'lingual' and type2 == 'lingual':
            # Lingual-to-lingual: dashed lines (linguistic connections)
            return {'horizontal': '┄', 'vertical': '┆', 'strength': 'medium'}
        elif (type1 == 'memory' and type2 == 'lingual') or (type1 == 'lingual' and type2 == 'memory'):
            # Cross-type: dotted lines (semantic bridges)
            return {'horizontal': '┈', 'vertical': '┊', 'strength': 'weak'}
        else:
            # Default: thin lines
            return {'horizontal': '─', 'vertical': '│', 'strength': 'default'}
    
    def _draw_connection_line(self, field, x1, y1, x2, y2, style):
        """Draw connection line with specified style"""
        # Simple line drawing using Bresenham-like algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            return
        
        # Use style-specific line characters
        if dx > dy:
            # More horizontal
            step_x = 1 if x2 > x1 else -1
            for x in range(x1, x2, step_x):
                y = y1 + int((x - x1) * (y2 - y1) / (x2 - x1))
                if 0 <= x < self.field_width and 0 <= y < self.field_height:
                    if field[y][x] == ' ':  # Don't overwrite particles
                        field[y][x] = style['horizontal']
        else:
            # More vertical
            step_y = 1 if y2 > y1 else -1
            for y in range(y1, y2, step_y):
                x = x1 + int((y - y1) * (x2 - x1) / (y2 - y1))
                if 0 <= x < self.field_width and 0 <= y < self.field_height:
                    if field[y][x] == ' ':  # Don't overwrite particles
                        field[y][x] = style['vertical']
    
    def _field_to_string(self, field):
        """Convert 2D field array to formatted string with type-aware legend"""
        particle_count = len(self.render_cache)
        
        # Count particles by type for header stats
        type_counts = {}
        for render_data in self.render_cache.values():
            particle_type = render_data.get('type', 'unknown')
            type_counts[particle_type] = type_counts.get(particle_type, 0) + 1
        
        # Header with type-specific statistics
        header = f"🧠 Quantum Particle Field - {particle_count} active particles\n"
        if type_counts:
            type_stats = " | ".join([f"{count} {ptype}" for ptype, count in type_counts.items()])
            header += f"   {type_stats}\n"
        header += "┌" + "─" * (self.field_width) + "┐\n"
        
        # Field content
        content = ""
        for row in field:
            content += "│" + "".join(row) + "│\n"
        
        # Enhanced footer with type-specific legend
        footer = "└" + "─" * (self.field_width) + "┘\n"
        footer += "Particle Types:\n"
        footer += "  Lingual: ◌◦○◎⊙⊚⬡ (communication/language)  Memory: ·▪■▬▲♦⬢ (storage/recall)\n"
        footer += "Quantum States: Superposition ◌▫  Uncertain ◦▪  Certain ⊙▲  Collapsed ⬡⬢\n"
        footer += "Connections: ═║ Memory-Memory  ┄┆ Lingual-Lingual  ┈┊ Cross-Type"
        
        return header + content + footer
    
    def build_field_visualization(self, particles, connections):
        # Your ASCII art generation logic here
        # Use thin lines (─│┌┐└┘├┤┬┴┼) for connections
        # Use particles as main focal points

      

        pass
