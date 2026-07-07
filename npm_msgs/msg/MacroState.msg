# MacroState.msg  — coordinator status / logging
uint8 PHASE_IDLE=0
uint8 PHASE_INFER=1
uint8 PHASE_PUSH=2
uint8 PHASE_SETTLE=3
Header   header
uint8    phase                          # 0 IDLE 1 INFER 2 PUSH 3 SETTLE
float32  dist_to_goal                   # m, object to world origin
bool     in_success_circle              # dist_to_goal < success_radius (0.2 m)
