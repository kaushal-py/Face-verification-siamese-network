#!/bin/sh

for f in PRE*.jpg;
do
	`mkdir Person-${f:4:7}`
	`mv *${f:4:7}* -t Person-${f:4:7}/`
done
