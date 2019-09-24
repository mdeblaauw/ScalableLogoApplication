#!/bin/bash

echo Upload webiste to logoapplication-website-39bk35l8 bucket.
aws s3 sync website/ s3://logoapplication-website-39bk35l8/
echo Upload finished.
