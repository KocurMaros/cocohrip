import rclpy
from rclpy.node import Node
import pygame
from checkers_game_tutorial.checkers.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED, WHITE
from checkers_game_tutorial.checkers.game import Game
from checkers_game_tutorial.minimax.algorithm import minimax
from checkers_msgs.msg import Board, Piece, Move

class CheckersNode(Node):
    def __init__(self):
        super().__init__('checkers_game')
        self.declare_parameter('fps', 60)
        self.FPS = self.get_parameter('fps').get_parameter_value().integer_value

        pygame.init()
        self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Checkers")

        self.game = Game(self.WIN)
        self.board_publisher = self.create_publisher(Board, 'board_topic', 10)  # Board publisher
        self.move_publisher = self.create_publisher(Move, 'move_topic', 10)  # Board publisher
        self.run_game()

    def get_row_col_from_mouse(self, pos):
        x, y = pos
        row = y // SQUARE_SIZE
        col = x // SQUARE_SIZE
        return row, col

    def color_to_string(self, color):
        if color == WHITE:
            return "white"
        elif color == RED:
            return "red"
        
    def publish_board_state(self):
        board_msg = Board()
        for piece in self.game.board.get_all_pieces():
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            # Set x and y if needed, depending on how your Piece class defines them
            board_msg.pieces.append(piece_msg)
        self.board_publisher.publish(board_msg)

    def publish_move_state(self, new_move, new_piece, removed):
        move_msg = Move()
        move_msg.target_row = new_move[0]
        move_msg.target_col = new_move[1]

        piece_msg = Piece()
        piece_msg.row = new_piece.row
        piece_msg.col = new_piece.col
        piece_msg.color = self.color_to_string(new_piece.color)
        piece_msg.king = new_piece.king
        move_msg.piece_for_moving = piece_msg
        
        # print(type(new_piece))
        for piece in removed:
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            # Set x and y if needed, depending on how your Piece class defines them
            move_msg.removed_pieces.append(piece_msg)

        self.move_publisher.publish(move_msg)

    def run_game(self):
        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(self.FPS)

            if self.game.turn == WHITE:
                self.publish_board_state()
                value, new_board, new_move, new_piece, removed = minimax(self.game.get_board(), 3, WHITE, self.game)
                self.game.ai_move(new_board)
                self.publish_move_state(new_move, new_piece, removed)

            if self.game.winner() != None:
                self.get_logger().info('Winner: {}'.format(self.game.winner()))
                run = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row, col = self.get_row_col_from_mouse(pos)
                    self.game.select(row, col)

            self.game.update()

        pygame.quit()

def main(args=None):
    rclpy.init(args=args)
    checkers_node = CheckersNode()
    rclpy.spin(checkers_node)

    checkers_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
