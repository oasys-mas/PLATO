
## Overview

This code provides 15 predefined wildfire settings organized into three categories based on task openness - whether new fires can appear during the episode.

## settings

### No Task Openness (5 settings)
Fires don't spread or appear randomly during the episode.

- `1fire_no_to`: Single fire in bottom middle (intensity 3)
- `2fire_no_to`: Two fires - bottom middle (intensity 3) + bottom right (intensity 1)
- `3fire_no_to`: Three fires - bottom middle (intensity 3) + both bottom corners (intensity 1)
- `2fire_mixed_no_to`: Two fires with different types - bottom middle (intensity 3, type 3) + bottom right (intensity 1, type 2)
- `2fire_hard_no_to`: Two difficult fires - bottom middle (intensity 3, type 3) + bottom right (intensity 1, type 3)

### Exogenous Task Openness (5 settings)
New fires can appear via random ignition (40% probability).

- `1fire_exo_to`, `2fire_exo_to`, `3fire_exo_to`, `2fire_mixed_exo_to`, `2fire_hard_exo_to`

### Endogenous Task Openness (5 settings)  
New fires can appear via fire spread from existing fires (40% probability).

- `1fire_endo_to`, `2fire_endo_to`, `3fire_endo_to`, `2fire_mixed_endo_to`, `2fire_hard_endo_to`



## Configuration Details

### Standard Settings (All settings)
- **Grid size**: Typically 3×2 (height × width)
- **Agent positions**: Top row, initial suppressant level = 3
- **Initial fuel**: 2 for all cells
- **Default fire type**: 2 (affects extinguishing difficulty)
- **Base reward**: 2
- **Reward formula**: `base_reward^fire_type` (only for cells with active fires)
- **Burnout probability**: 0.5 (fires can naturally burn out)
- **Burnout penalty**: Scaled = True (penalty = -1 × fire reward when attacking burned out fires)
- **No termination reward**

### Fire Behavior by Category
- **No Task Openness**: No fire spread, no random ignition
- **Exogenous Task Openness**: Random fire ignition (40% prob), no spread between fires
- **Endogenous Task Openness**: Fire spread between cells (40% prob), no random ignition

## Architecture

The system uses a clean object-oriented design:

- **`Wildfiresetting`**: Abstract base class defining the interface
- **Concrete setting classes**: Each handles one specific fire layout
- **Registry pattern**: Eliminates if-else statements, makes adding new settings simple
- **Inheritance hierarchy**: Shared behavior in base classes, specialization in derived classes

