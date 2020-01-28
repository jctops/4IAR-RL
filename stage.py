class Stage():
    def __init__(self, players, game, display=None):
        self.players = players
        self.game = game
        self.display = display
    
    def execute(self):
        state = self.game.get_initial_state()
        current_player = 1

        if self.display is not None:
            self.display.display_state(state)

        while not self.game.get_is_terminal_state(state):
            canonical_state = self.game.get_canonical_form(state, current_player)
            action = self.players[current_player - 1].get_action(canonical_state)
            print(action)
            assert self.game.get_allowed_actions(state, current_player)[action]
            state, current_player = self.game.get_next_state(state, current_player, action)
            if self.display is not None:
                self.display.display_state(state)
            
        print('Result!')
        print(self.game.get_result(state, current_player))