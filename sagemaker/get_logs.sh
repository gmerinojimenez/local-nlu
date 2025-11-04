#!/bin/bash

# Get CloudWatch logs for SageMaker training job
# Usage: ./sagemaker/get_logs.sh [JOB_NAME]

AWS_PROFILE="515422921637_AWSAdministratorAccess"
AWS_REGION="us-east-1"

if [ -z "$1" ]; then
    if [ -f /tmp/last_sagemaker_job.txt ]; then
        JOB_NAME=$(cat /tmp/last_sagemaker_job.txt)
    else
        echo "Error: Please provide job name"
        exit 1
    fi
else
    # Extract job name from ARN if full ARN is provided
    if [[ "$1" == arn:aws:sagemaker:* ]]; then
        JOB_NAME=$(echo "$1" | sed 's/.*training-job\///')
    else
        JOB_NAME=$1
    fi
fi

echo "Fetching logs for job: $JOB_NAME"
echo ""

# Get log stream name
LOG_GROUP="/aws/sagemaker/TrainingJobs"

echo "Searching for log streams..."
LOG_STREAMS=$(aws logs describe-log-streams \
    --log-group-name "$LOG_GROUP" \
    --log-stream-name-prefix "$JOB_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query 'logStreams[*].logStreamName' \
    --output text 2>/dev/null)

if [ -z "$LOG_STREAMS" ]; then
    echo "No log streams found for job: $JOB_NAME"
    echo "The job may not have started yet, or logs may not be available."
    exit 1
fi

echo "Found log streams:"
echo "$LOG_STREAMS"
echo ""
echo "=========================================="
echo "TRAINING JOB LOGS"
echo "=========================================="

# Get logs from all streams
for STREAM in $LOG_STREAMS; do
    echo ""
    echo "--- Log Stream: $STREAM ---"
    echo ""

    aws logs get-log-events \
        --log-group-name "$LOG_GROUP" \
        --log-stream-name "$STREAM" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'events[*].message' \
        --output text 2>/dev/null | tail -100
done

echo ""
echo "=========================================="
echo "END OF LOGS"
echo "=========================================="
