#!/usr/bin/env perl
use warnings;
use strict;

my ($filenames, $tag) = @ARGV;
die "usage: $0 <list of all training file names> <tag>\n" unless @ARGV;

my @files;
open(FP, "<", $filenames) or die "could not open $filenames: $!\n";

while(<FP>) {
	my $line = $_;
	chomp $line;
	push(@files, $line);

}
close FP;

my $outfile = "$tag\_TrainLog.tsv";
my $rep_count = 1;
open(OUT, ">", $outfile) or die "could not open $outfile: $!\n";

my $header = "Epoch\tTrainAUC\tDevAUC\tCost\tRep\n";	
print OUT $header;
foreach my $file (@files) {
	open(FILE, "<", $file);

	my @trains;
	my @devs;
	my @costs;
	my $skip_flag = 1; 
	while (<FILE>) {
		my $line = $_;
		chomp $line;
		
		if ($line =~ /Dev AUC/) {

			#warn "line is [$line]";

			my ($val) = $line =~ /: (.+)$/;
			push(@devs, $val);
			$skip_flag = 0;
		} elsif ($line =~ /Train AUC/) {
			my ($val) = $line =~ /: (.+)$/;
			push(@trains, $val);
		} elsif ($line =~ /Cost/) {
			if ($skip_flag == 1) {
				next;
			}
	
			my ($val) = $line =~ /: (.+)$/;
			push(@costs, $val);
			$skip_flag = 1;
		}
	}
	close(FILE);

	#warn "devs is " . @devs;


	for (my $epoch = 0; $epoch < @devs; $epoch++) {
		my $str = "$epoch\t$trains[$epoch]\t$devs[$epoch]\t$costs[$epoch]\t$rep_count\n";
		print OUT $str;
	}

	$rep_count++;
}
close OUT;
