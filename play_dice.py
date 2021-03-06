from colorama import init, Fore, Back, Style
init(autoreset=True)
print(Fore.GREEN + 'Initializing, please wait...')

import dice
import random
from numpy import ceil, floor
from time import sleep

played = False

capital = 100

def unit_s(amount):
	assert(type(amount) == int and amount > -1)
	if amount == 1:
		return 'unit'
	else:
		return 'units'
		
def take_bet(capital, played=True, second_bet=False):
	while True:
		if second_bet:
			message = 'By how much do you want to raise your current bet?\n(enter an integer amount, enter 0 if you don\'t want to raise): '
		else:
			message = 'How much do you want to bet? (enter an integer amount): '
		
		bet = input(message)
		
		if bet.isdigit():
			bet = int(bet)
			if bet > capital:
				print('You can\'t afford that honey. I\'ll ask again...')
				continue
			if bet == 0 and not second_bet:
				print('You must bet a nonzero amount in order to play. Let\'s try again...')
				continue
			else:
				break
		else:
			print('Invalid input. I\'ll ask again...')
			continue
			
	return bet

def show_capital(capital):
	print('You have ' + Fore.GREEN + Style.BRIGHT + str(capital) + Style.RESET_ALL + ' ' + unit_s(capital) + ' to spare.')

def show_hands(hand_p, hand_c):
	print('        Your hand : ' + checkered(hand_p))
	print('  =================================')
	print('  Computer\'s hand : ' + checkered(hand_c))

def checkered(s):
	checkered_s = []
	for i in range(len(s)):
		if i == 0:
			beginning = ''
		else:
			beginning = ' + '

		if i%2 == 0:
			checkered_s.append(str(Back.WHITE + Fore.BLACK + ' ' + s[i] + ' ' + Style.RESET_ALL))
		else:
			checkered_s.append(' ' + s[i] + ' ')
	return ''.join(checkered_s)

delay = 0.5

def evaluate(hand):
	occupants, core = dice.decon(hand)
	return dice.evaluate(occupants, core)

# Game procedure

while True:
	if capital <= 0:
		print('Get your broke ass off the poker table... adios!!')
		break

	if played:
		q = 'Play again? (y/n): '
	else:
		q = 'Want to play dice poker? (y/n): '
		
	play = input(q).lower()
	if play == 'n':
		break
	elif play != 'y':
		print('Invalid input. Please try again.')
		continue

	# Round 1 of the game

	print('\n ' + Back.GREEN + Fore.BLACK + '------------- Round 1 -------------' + Style.RESET_ALL + '\n')
	sleep(delay)
	print(Fore.GREEN + '<<Round 1 betting begins>>')

	if not played:
		show_capital(capital)
		
	played = True
	pool = 0

	# Take bet for round 1

	sleep(delay)
	bet = take_bet(capital, played=played)
	sleep(delay)
	print('Computer has matched your bet.')
	capital -= bet
	pool += 2 * bet

	sleep(delay)
	print(Fore.GREEN + '<<Round 1 betting ends>>')
	
	# Roll dice for round 1

	sleep(delay)
	print(Fore.GREEN + '<<Rolling dices>>')

	hand_p = dice.roll()
	hand_c = dice.roll()
	
	sleep(delay)
	show_hands(hand_p, hand_c)

	# Round 2 of the game.

	sleep(delay)
	print('\n ' + Back.GREEN + Fore.BLACK + '------------- Round 2 -------------' + Style.RESET_ALL + '\n')

	show_capital(capital)

	# Take bet for round 2

	sleep(delay)
	print(Fore.GREEN + '<<Round 2 betting begins>>')
	
	if capital > 0:
		bet_raise = take_bet(capital, second_bet=True)
		sleep(delay)
		print('Computer has matched your bet.')
		capital -= bet_raise
		pool += 2 * bet_raise

	# Find the optimum move for computer and raise bet if potentially profitable for computer (round 2)
		
	opt_c = dice.optimize(hand_c, hand_p)
	to_roll_c, p_loss_c = opt_c['Roll'], opt_c['P_loss']

	#print('Maximum probability of winning: ' + str(p_loss_c))

	if capital > 0:
		if p_loss_c < 0.5:
			if capital <= 10:
				to_raise_c = capital
			else:
				to_raise_c = int(floor(capital * 2 * (0.5 - p_loss_c)))
		
			if to_raise_c > 0:
				sleep(delay)
				while True:
					lost = True
					print('Computer has raised the bet by ' + Fore.GREEN + Style.BRIGHT + str(to_raise_c) + Style.RESET_ALL + ' units.')
					lion = input('Are you willing to match? Saying \'n\' means forfeiting your current bet (y/n): ').lower()
					if lion == 'n':
						print('You chicken! You have ' + Back.RED + Fore.WHITE + Style.BRIGHT + 'LOST' + Style.RESET_ALL + '.')
						show_capital(capital)
						break
					elif lion == 'y':
						lost = False
						capital -= to_raise_c
						pool += 2 * to_raise_c
						break
					else:
						print('Invalid input, please try again...')
						sleep(delay)
						continue
				if lost:
					continue

	sleep(delay)
	print(Fore.GREEN + '<<Round 2 betting ends>>')

	# Take input from player about rolling dices (round 2)

	while True:
		sleep(delay)
		to_roll_p = input('Which dices do you want to roll? \n(Enter dice positions without spaces, e.g., if you want to roll the first and the third dice, enter 13. If you don\'t want to roll any dice at all just press enter.): ')
		if to_roll_p.isdigit():
			to_roll_p = [int(c)-1 for c in to_roll_p]
			if min(to_roll_p) < 0 or max(to_roll_p) > 4:
				print('Dice position must be between 1 and 5 (inclusive), try again...')
				continue
			else:
				break
		elif to_roll_p == '':
			break
		else:
			print('Invalid input, try again...')
			continue

	# Roll dices (round 2)

	sleep(delay)
	print(Fore.GREEN + '<<Rolling dices>>')

	# Roll dices for player (round 2)

	new_dices = [str(random.randint(1,6)) for i in range(len(to_roll_p))]
	hand_temp = [c for c in hand_p]
	for i in range(len(to_roll_p)):
		hand_temp[to_roll_p[i]] = new_dices[i]
	hand_p = ''.join(hand_temp)

	# Roll dices for computer (round 2)
	
	sleep(delay)
	print('Computer is rolling: ' + ''.join([str(i+1) for i in to_roll_c]) + '\n')
	new_dices = [str(random.randint(1,6)) for i in range(len(to_roll_c))]
	hand_temp = [c for c in hand_c]
	for i in range(len(to_roll_c)):
		hand_temp[to_roll_c[i]] = new_dices[i]
	hand_c = ''.join(hand_temp)
	
	sleep(delay)
	show_hands(hand_p, hand_c)

	# Evaluate scores and settle bets

	sleep(delay)
	print('\n ' + Fore.BLACK + Back.GREEN + '------------- Results -------------' + Style.RESET_ALL + '\n')
	
	state_p, value_p = evaluate(hand_p)
	state_c, value_c = evaluate(hand_c)
	sleep(delay)
	print('      Your score: ' + state_p + ' (Rank: ' + str(dice.state_rank[state_p]) + ', Sum: ' + str(value_p) + ')')
	print('Computer\'s score: ' + state_c + ' (Rank: ' + str(dice.state_rank[state_c]) + ', Sum: ' + str(value_c) + ')')
	
	sleep(delay)
	score_p, score_c = dice.rank_f[hand_p], dice.rank_f[hand_c]
	if score_p > score_c:
		print('You have ' + Fore.RED + Style.BRIGHT + 'WON' + Style.RESET_ALL + '.')
		capital += pool
		show_capital(capital)
	elif score_p == score_c:
		print('The game was a ' + Fore.RED + Style.BRIGHT + 'TIE' + Style.RESET_ALL + '.')
		capital += int(pool/2)
		show_capital(capital)
	else:
		print('You have ' + Fore.RED + Style.BRIGHT + 'LOST' + Style.RESET_ALL + '.')
		show_capital(capital)
		
	
	
