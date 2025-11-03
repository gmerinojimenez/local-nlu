# AWS SageMaker Training Guide

This guide helps you train your NLU model on AWS SageMaker instead of locally.

## Prerequisites

1. **AWS CLI installed and configured**
   ```bash
   aws --version
   aws configure --profile 515422921637_AWSAdministratorAccess
   ```

2. **SageMaker Execution Role**
   - You need a SageMaker execution role ARN
   - Update `SAGEMAKER_ROLE` in `launch_sagemaker_job.sh`
   - Find it in: AWS Console → SageMaker → Roles
   - Or create one with: `aws iam create-role --role-name SageMakerExecutionRole ...`

3. **Processed Data**
   - Make sure you have run: `python scripts/preprocess_data.py`
   - This creates: `data/processed/train.csv`, `val.csv`, `test.csv`

## Quick Start

### Step 1: Make scripts executable
```bash
chmod +x sagemaker/launch_sagemaker_job.sh
chmod +x sagemaker/monitor_job.sh
```

### Step 2: Update configuration

Edit `sagemaker/launch_sagemaker_job.sh`:
- Set `AWS_REGION` (default: us-east-1)
- Set `SAGEMAKER_ROLE` (your SageMaker execution role ARN)
- Choose `INSTANCE_TYPE`:
  - `ml.g4dn.xlarge` - GPU (recommended, ~$0.70/hour)
  - `ml.p3.2xlarge` - Faster GPU (~$3.06/hour)
  - `ml.m5.2xlarge` - CPU only (~$0.46/hour, slower)

### Step 3: Launch training job
```bash
./sagemaker/launch_sagemaker_job.sh
```

This will:
1. Create S3 bucket for data and outputs
2. Upload training data (train.csv, val.csv, test.csv)
3. Package and upload source code
4. Launch SageMaker training job

### Step 4: Monitor training
```bash
./sagemaker/monitor_job.sh
```

Or view in AWS Console:
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs

### Step 5: Download trained model

After training completes:
```bash
# The monitor script will show you the command, or:
aws s3 sync s3://YOUR-BUCKET/output/JOB_NAME/output/ ./models/sagemaker_model/ \
    --profile 515422921637_AWSAdministratorAccess
```

Then test it:
```bash
python scripts/test_checkpoint.py models/sagemaker_model/best_model
```

## Training Configuration

### Hyperparameters

Edit in `launch_sagemaker_job.sh` (line ~90):
```json
"HyperParameters": {
    "model-name": "t5-base",        # or "t5-small" for faster training
    "num-epochs": "15",              # increase for better quality
    "batch-size": "16",              # reduce if GPU memory issues
    "learning-rate": "0.00002"
}
```

### Instance Types

| Instance | GPU | vCPU | RAM | Price/hour | Training Time (15 epochs) |
|----------|-----|------|-----|------------|---------------------------|
| ml.g4dn.xlarge | 1x T4 | 4 | 16 GB | $0.70 | ~4-6 hours |
| ml.p3.2xlarge | 1x V100 | 8 | 61 GB | $3.06 | ~2-3 hours |
| ml.m5.2xlarge | None | 8 | 32 GB | $0.46 | ~20-30 hours |

**Recommendation**: Use `ml.g4dn.xlarge` (good balance of speed and cost)

## Cost Estimation

For 15 epochs with `ml.g4dn.xlarge`:
- Training time: ~5 hours
- Cost: $0.70/hour × 5 hours = **~$3.50**
- Storage (S3): ~$0.10 for data storage
- **Total: ~$3.60**

## Monitoring Commands

```bash
# Check job status
aws sagemaker describe-training-job \
    --training-job-name YOUR_JOB_NAME \
    --profile 515422921637_AWSAdministratorAccess

# View CloudWatch logs
aws sagemaker describe-training-job \
    --training-job-name YOUR_JOB_NAME \
    --profile 515422921637_AWSAdministratorAccess \
    --query 'TrainingJobStatus'

# List all training jobs
aws sagemaker list-training-jobs \
    --profile 515422921637_AWSAdministratorAccess \
    --max-results 10
```

## Troubleshooting

### Error: "Role ARN not found"
- Update `SAGEMAKER_ROLE` in `launch_sagemaker_job.sh`
- Create role: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html

### Error: "ResourceLimitExceeded"
- Your AWS account has instance limits
- Request quota increase or use different instance type

### Error: "Access Denied to S3"
- Ensure SageMaker execution role has S3 access
- Add policy: `AmazonS3FullAccess` to the role

### Training fails with OOM (Out of Memory)
- Reduce `batch-size` to 8 or 4
- Or use larger instance: `ml.p3.2xlarge`

## Cleanup

After training, clean up resources:
```bash
# Delete S3 bucket (replace with your bucket name)
aws s3 rb s3://YOUR-BUCKET --force \
    --profile 515422921637_AWSAdministratorAccess
```

## Advanced: Resume Training

To resume from a checkpoint, modify the script to:
1. Upload existing checkpoint to S3
2. Add `--checkpoint-dir` parameter to training script
3. Load from checkpoint in `train_sagemaker.py`

## Support

For issues:
- Check CloudWatch logs in AWS Console
- Review SageMaker job details
- Verify IAM permissions for SageMaker execution role
