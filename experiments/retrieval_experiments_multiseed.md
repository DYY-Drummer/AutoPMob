# Multi-seed Retrieval Experiment Summary

Seeds: [42, 123, 456, 789, 1024]

## Test Metrics (mean ± std, 95% CI)

| Method | MRR | R@1 | R@5 | R@10 |
|--------|-----|-----|-----|------|
| baseline_mix | 0.890±0.095 [0.804, 0.949] | 0.826±0.154 | 0.975±0.048 | 0.987±0.030 |
| svd_cos | 0.667±0.140 [0.564, 0.771] | 0.523±0.168 | 0.874±0.105 | 0.944±0.056 |

## Test Metrics by Variant Type

| Method | Variant | MRR (mean±std) | Avg n |
|--------|---------|----------------|-------|
| baseline_mix | context_paraphrased | 0.864±0.098 | 78 |
| baseline_mix | original | 0.851±0.080 | 26 |
| baseline_mix | random_io_from_models | 0.924±0.112 | 104 |
| baseline_mix | swap_io | 0.866±0.094 | 26 |
| svd_cos | context_paraphrased | 0.667±0.145 | 78 |
| svd_cos | original | 0.686±0.155 | 26 |
| svd_cos | random_io_from_models | 0.662±0.134 | 104 |
| svd_cos | swap_io | 0.665±0.148 | 26 |

## Per-seed Details

### Seed 42 (train=612, val=729, test=45)

- **baseline_mix**: val MRR=0.9533, test MRR=0.8993
  - context_paraphrased: MRR=0.8196 (n=15)
  - original: MRR=0.8182 (n=5)
  - random_io_from_models: MRR=1.0000 (n=20)
  - swap_io: MRR=0.8167 (n=5)
- **svd_cos**: val MRR=0.7621, test MRR=0.7741
  - context_paraphrased: MRR=0.7944 (n=15)
  - original: MRR=0.8667 (n=5)
  - random_io_from_models: MRR=0.7375 (n=20)
  - swap_io: MRR=0.7667 (n=5)

### Seed 123 (train=468, val=315, test=603)

- **baseline_mix**: val MRR=0.9450, test MRR=0.9391
  - context_paraphrased: MRR=0.9229 (n=201)
  - original: MRR=0.9058 (n=67)
  - random_io_from_models: MRR=0.9627 (n=268)
  - swap_io: MRR=0.9271 (n=67)
- **svd_cos**: val MRR=0.6046, test MRR=0.8218
  - context_paraphrased: MRR=0.8229 (n=201)
  - original: MRR=0.8214 (n=67)
  - random_io_from_models: MRR=0.8132 (n=268)
  - swap_io: MRR=0.8528 (n=67)

### Seed 456 (train=1080, val=180, test=126)

- **baseline_mix**: val MRR=0.9169, test MRR=0.7265
  - context_paraphrased: MRR=0.7210 (n=42)
  - original: MRR=0.7381 (n=14)
  - random_io_from_models: MRR=0.7262 (n=56)
  - swap_io: MRR=0.7321 (n=14)
- **svd_cos**: val MRR=0.6703, test MRR=0.4678
  - context_paraphrased: MRR=0.4673 (n=42)
  - original: MRR=0.4930 (n=14)
  - random_io_from_models: MRR=0.4605 (n=56)
  - swap_io: MRR=0.4730 (n=14)

### Seed 789 (train=1071, val=81, test=234)

- **baseline_mix**: val MRR=0.7305, test MRR=0.9147
  - context_paraphrased: MRR=0.8848 (n=78)
  - original: MRR=0.8487 (n=26)
  - random_io_from_models: MRR=0.9615 (n=104)
  - swap_io: MRR=0.8827 (n=26)
- **svd_cos**: val MRR=0.2328, test MRR=0.6548
  - context_paraphrased: MRR=0.6313 (n=78)
  - original: MRR=0.6395 (n=26)
  - random_io_from_models: MRR=0.6837 (n=104)
  - swap_io: MRR=0.6248 (n=26)

### Seed 1024 (train=1017, val=207, test=162)

- **baseline_mix**: val MRR=0.9119, test MRR=0.9691
  - context_paraphrased: MRR=0.9722 (n=54)
  - original: MRR=0.9444 (n=18)
  - random_io_from_models: MRR=0.9722 (n=72)
  - swap_io: MRR=0.9722 (n=18)
- **svd_cos**: val MRR=0.7674, test MRR=0.6149
  - context_paraphrased: MRR=0.6174 (n=54)
  - original: MRR=0.6101 (n=18)
  - random_io_from_models: MRR=0.6160 (n=72)
  - swap_io: MRR=0.6080 (n=18)

