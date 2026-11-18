# Gym member segments: who should get the monthly offer?

TECH-UB 24 individual clustering assignment (k-means). Avirul Islam.

## Reproduce
```bash
pip install -r requirements.txt
python analysis.py
```
All numbers on the slides come from `outputs/results.json`, `outputs/segment_profile.csv` and `outputs/elbow_table.csv`. Running the script also writes the elbow chart (`outputs/elbow_train_test.png`) and every member's segment (`outputs/members_with_segments.csv`).

## Method
- **Data:** `data/gym_members.csv`, 330 members, averages over the last 3 months.
- **Split:** 70/30 random split (seed 42): 231 training, 99 test members.
- **Features:** age, visits per month, minutes per visit, classes per month, months as member, monthly extras spend. `member_id` left out (an identifier, not a trait).
- **Scaling:** standardized (z-scores), with the scaler fit on training members only.
- **Model:** k-means, k = 1 to 8 on training members; k chosen from the elbow. k = 1 is the baseline.
- **Test check:** test members are assigned to the training centers (no refit) and their spread is summed for each k.
- **Decision:** a segment gets the $10 credit if, on training members, giving it the credit nets the gym more than $0 under finance's assumptions (member by member).
- **Value:** monthly net from the credit for all 330 members, baseline (credit to everyone) vs v1 (credit to chosen segments only), applied member by member.
- **Robustness:** clustering without extras spend assigns 99.7% of members to the same segments.
