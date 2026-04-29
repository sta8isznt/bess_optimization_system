# Energy Optimization System

## Steps:

### PHASE 1: HANDLING DATA SCARCIRY 

* Ανάλυση του Oxford Battery Degradation Dataset

> * Ανάγνωση και καθαρισμός (μέσω Python/Pandas) των αρχείων CSV του dataset της Οξφόρδης. Υπολογισμός στατιστικών συσχετίσεων μεταξύ των προφίλ εμπορίας (arbitrage) και της πτώσης της χωρητικότητας (capacity fade) ανά κύκλο.

> * Έξοδος (Output): Εξαγόμενα χαρακτηριστικά (Features): Βάθος Εκφόρτισης (DoD), C-rate, και η αντίστοιχη πτώση του State of Health (SOH).

* Προσομοίωση Φυσικού Μοντέλου (PyBaMM - SPM)

> * Στήσιμο ενός Single Particle Model (SPM).

> * Είσοδος : Τα χαρακτηριστικά του Βήματος 1 + Τεχνικές προδιαγραφές μπαταρίας της Metlen.

> * Έξοδος : Ένα ψηφιακό μοντέλο ικανό να υπολογίσει τη φυσική καταπόνηση για οποιοδήποτε μελλοντικό προφίλ φόρτισης/εκφόρτισης του δώσουμε.

* Δημιουργία Look-Up Table (LUT) Κόστους Φθοράς:

Τρέχουμε το PyBaMM (Βήμα 2) offline για χιλιάδες πιθανά σενάρια Βάθους Εκφόρτισης (DoD). Μεταφράζουμε τη φυσική φθορά σε ευρώ, διαιρώντας την απώλεια ζωής με το CAPEX της μπαταρίας. Αποθηκεύουμε τα αποτελέσματα σε έναν πίνακα (LUT).

> * Είσοδος: Τα αποτελέσματα προσομοίωσης του PyBaMM + Κόστος αντικατάστασης μπαταρίας (€/MW).

> * Έξοδος: Ένας πίνακας (DoD, Θερμοκρασία) -> Marginal Degradation Cost (€/MWh).

#### PyBaMM LUT Generation (Digital Twin Physical Layer)

Use the physical layer script to run SPM + thermal + degradation simulations and create a LUT:

```bash
python scripts/generate_lut.py --dod-range 0.1,1.0,0.1 --temp-range 10,45,5 --output lut_dod_temp.csv
```

This produces a LUT in `data/processed/` with columns:
`dod`, `temperature_c`, `soh_start`, `soh_end`, `soh_drop`, `energy_mwh`, `v_deg_eur_per_mwh`.


### PHASE 2: OPTIMIZATION

* Ingestion Δεδομένων Αγοράς (HEnEx DAM) -> παραγωγή `price_signals_15m.csv`

* Διατύπωση Προβλήματος MILP (με εισόδους το `price_signals_15m.csv` και τις εντολές `B`,`S`,`H`)

* Γραμμικοποίηση της Φθοράς με SOS2 vars:  MNLIP -> MILP!!!

* Ενσωμάτωση Τεχνικών Περιορισμών -> περιορισμό διάρκειας συστήματος (π.χ. 2 ώρες max σε πλήρη ισχύ) και τα όρια υγείας (SoC{min,max})

> * Είσοδος: Οι ρυθμιστικοί κανόνες της αγοράς και του ΑΔΜΗΕ.

### PHASE 3: TEST & DELIVERABLE

* Backtesting και Υπολογισμός Εμπορικών Δεικτών (IRR)

* Dashboard (DAM vs Power of BESS, SoC Trajectory, Total profit vs Dedagration cost)















