# PushResult.msg 
Header   header
bool     contact_made
float32  peak_force                     # N, from ee wrench estimate 
float32  finger_travel                  # m, hand displacement during push 
string   end_reason                     # "reach" | "timeout" | "contact_lost" | "dry_run"
