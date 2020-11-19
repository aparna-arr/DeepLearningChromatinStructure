#!/usr/bin/env perl
use warnings;
use strict;

my ($loglist) = @ARGV;
die "usage: $0 <file with list of .log files>\n" unless @ARGV == 1;

my @logfiles;

open(LOGLIST, "<", $loglist) or die "could not open $loglist:$!\n";

while(<LOGLIST>) {
	my $line = $_;
	chomp $line;

	push(@logfiles,$line);
}

close LOGLIST;

my @results;

foreach my $file (@logfiles) {
	my $devROC = 0;
	my $trainROC = 0;

	warn "on file $file";

	my $model = $file;

	if ($file =~ /conv/) { 
		open(LOG, "<", $file) or die "could not open $file: $!\n";
	
		while(<LOG>) {
			my $line = $_;
			chomp $line;
			if ($line =~ /ROC/) {
				if ($line =~ /Dev/) {
					($devROC) = $line =~ /Dev.+: (\S+)$/;
				} else {
					($trainROC) = $line =~ /Train.+: (\S+)$/;
				}
			}
		}

		close LOG;

	} else {
		open(LOG, "<", $file) or die "could not open $file: $!\n";
	
		while(<LOG>) {
			my $line = $_;
			chomp $line;

			($trainROC, $devROC) = $line =~ /train AUC: (.+) dev AUC: (.+)$/ 
		}

		close LOG;
	}

	push(@results, {
		model => $model,
		trainROC => $trainROC,
		devROC => $devROC
	});	
}

warn "after reading in all files";

my @sortDevROC = sort{ $b->{devROC} <=> $a->{devROC} } @results;

warn "best dev ROC is $sortDevROC[0]{devROC} from model $sortDevROC[0]{model}";

# generate latex table

my $header = "\\begin{tabular}{||c|c|c||}\n\\hline\n";

$header .= "Model & Train Score & Dev Score \\\\\n\\hline\\hline\n";
foreach my $hashref (@sortDevROC){
	my $model = $hashref->{model};
	my $trainROC = $hashref->{trainROC};
	my $devROC = $hashref->{devROC};

	$header .= "$model & $trainROC & $devROC \\\\\n\\hline\n";
}

open(OUT, ">", "latex.txt") or die "could not open outfile!:$!\n";
print OUT $header;
close OUT;
